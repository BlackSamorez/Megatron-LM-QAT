import os
import threading
import queue
from time import perf_counter
import tempfile
import shutil
import threading
import sys
import time
import torch
import torch.distributed as dist

from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import InferenceWrapperConfig
from pretrain_gpt import model_provider, train_valid_test_datasets_provider, get_batch, tokens_to_packed_seq_params
from megatron.training import get_args, get_model, print_rank_0, get_timers, get_tokenizer
from megatron.training.initialize import initialize_megatron
from megatron.training.checkpointing import load_checkpoint
from megatron.core import mpu
from megatron.training.training import build_train_valid_test_data_iterators, update_train_iters
from megatron.training.global_vars import get_wandb_writer


# ---- 1. Robust Contiguity Tracker (Unchanged) ----
class ProgressTracker:
    """
    Tracks LOCAL progress relative to the CURRENT run.
    starting_step is usually 0 for a fresh process execution, 
    as we calculate absolute offsets separately.
    """
    def __init__(self):
        self.next_expected_step = 0
        self.completed_buffer = set()
        # -1 implies "I haven't finished step 0 yet"
        self.highest_contiguous_step = -1 

    def process_completion(self, step_idx):
        self.completed_buffer.add(step_idx)
        while self.next_expected_step in self.completed_buffer:
            self.completed_buffer.remove(self.next_expected_step)
            self.highest_contiguous_step = self.next_expected_step
            self.next_expected_step += 1
        return self.highest_contiguous_step


# ---- 2. Safe IO Worker (Now with fsync/Atomic Rename) ----
SAVER_WORKERS = 16

def _save_file_safe(chunk_id: int, payload: dict, dst_path: str):
    """
    Saves file with hard durability guarantees:
    1. Write to temp file
    2. Flush buffers
    3. fsync to disk
    4. Atomic rename to final destination
    """
    filename = os.path.join(dst_path, f"{chunk_id:010d}.pt")
    dirname = os.path.dirname(filename)
    
    # Create temp file in same directory to ensure atomic move works
    fd, tmp_path = tempfile.mkstemp(dir=dirname, prefix=f"tmp_{chunk_id}_")
    
    try:
        with os.fdopen(fd, 'wb') as f:
            torch.save(payload, f)
            f.flush()           # Flush Python buffers
            os.fsync(f.fileno()) # Force OS to write to physical disk
            
        # Atomic switch: The file instantly appears as valid data
        os.rename(tmp_path, filename)
        
    except Exception as e:
        # Cleanup garbage if we failed
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e

def _save_worker(processed_q: queue.Queue, status_q: queue.Queue, dst_path: str):
    from concurrent.futures import ThreadPoolExecutor
    
    sem = threading.BoundedSemaphore(value=SAVER_WORKERS)

    def _run_one(msg): 
        # msg: (relative_step_idx, absolute_chunk_id, payload)
        rel_step, chunk_id, payload = msg
        try:
            _save_file_safe(chunk_id, payload, dst_path)
            status_q.put((rel_step, None))
        except Exception as e:
            status_q.put((rel_step, e))
        finally:
            sem.release() 

    with ThreadPoolExecutor(max_workers=SAVER_WORKERS) as pool:
        while True:
            sem.acquire() 
            msg = processed_q.get()
            if msg is None: break
            pool.submit(_run_one, msg)


# ---- 3. Elastic Global Progress Sync ----

def save_global_progress_file(total_chunks: int, progress_file_name: str):
    """Writes the total number of chunks saved globally."""
    # Atomic write pattern
    dir_name = os.path.dirname(progress_file_name)
    with tempfile.NamedTemporaryFile('w', dir=dir_name, delete=False) as tmp:
        tmp.write(str(total_chunks))
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    shutil.move(tmp_name, progress_file_name)

def sync_and_save_elastic(tracker: ProgressTracker, 
                          accumulated_history: int, 
                          progress_file_name: str):
    """
    Calculates safe global progress even if World Size changed.
    Formula: Total = History + (Min(Current_Run_Steps) * Current_World_Size)
    """
    # 1. Get how many steps (batches) THIS rank has finished in THIS run
    local_safe_run_step = tracker.highest_contiguous_step
    
    # 2. If I'm not a saver (TP rank > 0), I am "infinitely done"
    if mpu.get_tensor_model_parallel_rank() != 0:
        local_safe_run_step = 999999999
        
    # 3. Find the lowest step completed by ALL savers in the current cluster
    # Note: Since highest_contiguous_step starts at -1, we add 1 for calculation 
    # to represent "count of completed steps"
    completed_steps = local_safe_run_step + 1
    
    t_tensor = torch.tensor([completed_steps], dtype=torch.long, device=torch.cuda.current_device())
    dist.all_reduce(t_tensor, op=dist.ReduceOp.MIN)
    global_min_run_steps = t_tensor.item()

    # 4. Calculate Total Chunks
    # We only update if we have completed at least 1 step in this run
    current_dp_size = mpu.get_data_parallel_world_size()
    
    # The new safe total is history + (batches_done_now * chunks_per_batch)
    new_global_total = accumulated_history + (global_min_run_steps * current_dp_size)
    
    if dist.get_rank() == 0:
        save_global_progress_file(new_global_total, progress_file_name)
        print_rank_0(f"Global Progress Updated: {new_global_total} total chunks saved.")
    
    return new_global_total


def drain_status_queue(status_q: queue.Queue, tracker: ProgressTracker):
    while not status_q.empty():
        try:
            step_idx, error = status_q.get_nowait()
            if error: raise error
            tracker.process_completion(step_idx)
        except queue.Empty:
            break

def add_logits_args(parser):
    group = parser.add_argument_group(title='logits generation')
    group.add_argument("--logits_save_dir", type=str, required=True, help='Directory to save logits.')
    group.add_argument("--logits_top_k", type=int, default=256, help='Top-K logits.')
    group.add_argument("--logits_samples", type=int, default=None, help='Total samples to save')
    group.add_argument("--logits_samples_per_chunk", type=int, default=32, help='Samples per chunk')
    group.add_argument("--logits_sync_iters", type=int, default=128, help='Blocking flush every N iters')
    return parser

# ---- Main ----

def main():
    initialize_megatron(
        extra_args_provider=add_logits_args,
        args_defaults={'exit_on_missing_checkpoint': True}
    )
    args = get_args()
    
    # Model Setup
    model = get_model(model_provider, wrap_with_ddp=False)
    load_checkpoint(model, None, None)
    model = model[0].module
    model.eval()
    model.model_is_pipeline_parallel = False
    tokenizer = get_tokenizer()

    timers = get_timers()
    wandb_writer = get_wandb_writer()

    if mpu.get_data_parallel_rank() == 0:
        os.makedirs(args.logits_save_dir, exist_ok=True)
    dist.barrier()
    
    # ---- Resume Logic (Chunk Based) ----
    progress_file = os.path.join(args.logits_save_dir, "progress_chunks.txt")
    accumulated_chunks = 0

    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                accumulated_chunks = int(f.read())
                print_rank_0(f"Resuming from history: {accumulated_chunks} chunks saved.")
        except:
            print_rank_0("Progress file error. Starting from 0.")
    else:
        if dist.get_rank() == 0:
             save_global_progress_file(0, progress_file)
             
    # Sync accumulated chunks across ranks (just in case)
    acc_tensor = torch.tensor([accumulated_chunks], dtype=torch.long, device=torch.cuda.current_device())
    dist.broadcast(acc_tensor, src=0)
    accumulated_chunks = acc_tensor.item()
    accumulated_chunks = (accumulated_chunks // mpu.get_data_parallel_world_size()) * mpu.get_data_parallel_world_size()

    # ---- Data Loader Setup ----
    # IMPORTANT! >>
    assert args.micro_batch_size == args.logits_samples_per_chunk, "Data offset assumes this"
    args.iteration = 0
    args.consumed_train_samples = accumulated_chunks * args.logits_samples_per_chunk
    args.train_samples = args.logits_samples
    update_train_iters(args)
    train_valid_test_datasets_provider.is_distributed = True
    # IMPORTANT! <<
    train_data_iterator, _, _ = build_train_valid_test_data_iterators(train_valid_test_datasets_provider)
    
    # ---- Saver Setup ----
    saver_payload_q, saver_status_q, save_worker, tracker = None, None, None, None
    
    if mpu.get_tensor_model_parallel_rank() == 0:
        saver_payload_q = queue.Queue(maxsize=8) 
        saver_status_q = queue.Queue(maxsize=32)
        
        save_worker = threading.Thread(
            target=_save_worker, 
            args=(saver_payload_q, saver_status_q, args.logits_save_dir), 
            daemon=True
        )
        save_worker.start()
        
        tracker = ProgressTracker()

    dist.barrier()
    
    # ---- Main Loop ----
    # We run a relative loop for this specific execution
    current_run_step = 0 
    
    print_rank_0(f"Starting run. Topology: {mpu.get_data_parallel_world_size()} DP workers.")
    global_progress = accumulated_chunks

    with torch.no_grad():        
        while global_progress * args.logits_samples_per_chunk < args.logits_samples:
            for _ in range(args.logits_sync_iters):
                start = perf_counter()
                timers('batch-generator', log_level=1).start()
                # Loader automatically skips 'accumulated_chunks' + 'current_run_step * world_size'
                tokens, labels, loss_mask, attention_mask, position_ids = get_batch(train_data_iterator)
                orig_seq_len = position_ids.size(1)
                position_ids = position_ids.view(1, -1)
                tokens = tokens.view(1, -1)
                labels = labels.view(1, -1)
                loss_mask = loss_mask.view(1, -1)
                packed_seq_params = tokens_to_packed_seq_params(tokens, tokenizer.eod, orig_seq_len)
                timers('batch-generator').stop()
                
                # Forward Pass
                teacher_probs, teacher_positions = model(
                    tokens, position_ids, attention_mask,
                    packed_seq_params=packed_seq_params,
                    runtime_gather_output=True,
                    return_topk=args.logits_top_k,
                )

                # --- Dispatch Save (Elastic ID) ---
                if mpu.get_tensor_model_parallel_rank() == 0:
                    
                    # ID CALCULATION: 
                    # History + (Steps in this run * Current World Size) + My Rank
                    chunk_id = accumulated_chunks + \
                            (current_run_step * mpu.get_data_parallel_world_size()) + \
                            mpu.get_data_parallel_rank()
                    
                    payload = {
                        "input_ids": tokens.to("cpu", non_blocking=False),
                        "labels": labels.to("cpu", non_blocking=False),
                        "exp_logits": teacher_probs.to("cpu", non_blocking=False),
                        "index": teacher_positions.to("cpu", non_blocking=False),
                        "loss_mask": loss_mask.to("cpu", non_blocking=False), 
                        "cu_seqlens": packed_seq_params.cu_seqlens_q.to("cpu", non_blocking=False),
                    }
                    
                    # We pass 'current_run_step' to tracker, but 'chunk_id' to saver
                    saver_payload_q.put((current_run_step, chunk_id, payload))
                    drain_status_queue(saver_status_q, tracker)

                # --- Periodic Sync ---
                if current_run_step % SAVER_WORKERS == 0:
                    if mpu.get_tensor_model_parallel_rank() == 0:
                        drain_status_queue(saver_status_q, tracker)
                    
                    # Syncs based on accumulated history + current safe progress
                    global_progress = sync_and_save_elastic(tracker if tracker else None, accumulated_chunks, progress_file)

                current_run_step += 1
                end = perf_counter()
                
                throughput = tokens.numel() / (end - start)
                if wandb_writer is not None:
                    wandb_writer.log({
                        'throughput': throughput,
                        'chunk_id': chunk_id,
                        'consumed_tokens': chunk_id * args.logits_samples_per_chunk * args.seq_length,
                        'global_progress': global_progress,
                    }, chunk_id)
                
                # bos_count = (tokens == 1).sum().cpu().item()
                # eos_count = (tokens == 2).sum().cpu().item()    
                # print_rank_0(f"{chunk_id}:\n\t{tokens[0, :10].cpu().tolist()=}\n\t{bos_count=}\n\t{eos_count=}\n\t{teacher_probs[:10, 0, 0].cpu().tolist()=}\n\t{teacher_probs[-10:, 0, 0].cpu().tolist()=}")
                print_rank_0(f"\ttok/s: {throughput:.0f}")
            
            # --- BLOCKING FLUSH (End of Inner Loop) ---
            print_rank_0(f"Syncing at step {current_run_step}...")
            
            # 1. Wait for local saver to finish EVERYTHING sent so far
            if mpu.get_tensor_model_parallel_rank() == 0:
                # We want tracker to catch up to the last step we submitted (current_run_step - 1)
                last_submitted_step = current_run_step - 1
                while tracker.highest_contiguous_step < last_submitted_step:
                    drain_status_queue(saver_status_q, tracker)
                    time.sleep(0.01) # Don't busy-loop the CPU

            # 2. Synchronize all ranks (everyone is now physically done)
            dist.barrier()

            # 3. Update Global Progress File
            global_progress = sync_and_save_elastic(tracker if tracker else None, accumulated_chunks, progress_file)
            print_rank_0(f"Block Done. Global Progress: {global_progress}")
            
    if mpu.get_tensor_model_parallel_rank() == 0:
        saver_payload_q.put(None)
        save_worker.join()
        drain_status_queue(saver_status_q, tracker)
    

    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
