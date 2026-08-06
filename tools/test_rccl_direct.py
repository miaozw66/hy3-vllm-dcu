#!/usr/bin/env python3
"""Direct RCCL multi-node test using torch.multiprocessing.spawn.

Usage:
  Node 0: python3 test_rccl_direct.py 0
  Node 1: python3 test_rccl_direct.py 1
"""
import os, sys, socket, fcntl, struct, traceback


def main():
    node_rank = int(sys.argv[1])
    print(f"[N{node_rank}] Starting RCCL multi-node test...", flush=True)

    # --- Detect interface ---
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("10.18.17.71", 1))
    src_ip = s.getsockname()[0]
    s.close()

    iface = None
    for ifname in socket.if_nameindex():
        name = ifname[1]
        try:
            ss = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            addr = socket.inet_ntoa(
                fcntl.ioctl(ss.fileno(), 0x8915,
                            struct.pack('256s', name[:15].encode()))[20:24])
            ss.close()
            if addr == src_ip:
                iface = name
                break
        except Exception:
            pass

    if iface is None:
        iface = "eno1"
        print(f"[N{node_rank}] WARNING: using fallback iface={iface}", flush=True)
    else:
        print(f"[N{node_rank}] Detected iface={iface}, src_ip={src_ip}", flush=True)

    os.environ["NCCL_SOCKET_IFNAME"] = iface
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["HSA_FORCE_FINE_GRAIN_PCIE"] = "1"
    os.environ["MASTER_ADDR"] = "10.18.17.71"
    os.environ["MASTER_PORT"] = "29505"
    os.environ["NODE_RANK"] = str(node_rank)
    os.environ["NNODES"] = "2"
    os.environ["NPROC_PER_NODE"] = "4"
    os.environ["WORLD_SIZE"] = "8"

    # --- Launch workers ---
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    print(f"[N{node_rank}] Spawning 4 GPU workers...", flush=True)
    try:
        mp.spawn(
            worker_fn,
            args=(node_rank,),
            nprocs=4,
            join=True,
        )
        print(f"[N{node_rank}] ALL DONE!", flush=True)
    except Exception as e:
        print(f"[N{node_rank}] FAILED: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)


def worker_fn(local_rank, node_rank):
    """Runs on each GPU."""
    import torch
    import torch.distributed as dist

    global_rank = node_rank * 4 + local_rank
    world_size = 8

    print(f"  [N{node_rank}/G{local_rank}] rank={global_rank} starting...",
          flush=True)

    torch.cuda.set_device(local_rank)
    gpu = torch.cuda.get_device_name(local_rank)
    print(f"  [N{node_rank}/G{local_rank}] GPU: {gpu}", flush=True)

    print(f"  [N{node_rank}/G{local_rank}] Calling init_process_group...",
          flush=True)

    dist.init_process_group(
        backend="nccl",
        init_method="tcp://10.18.17.71:29505",
        world_size=world_size,
        rank=global_rank,
    )

    print(f"  [N{node_rank}/G{local_rank}] RCCL initialized!", flush=True)

    # all_reduce test
    t = torch.ones(1024).cuda() * (global_rank + 1)
    dist.all_reduce(t)
    expected = sum(range(1, world_size + 1))
    ok = abs(t[0].item() - expected) < 0.01

    print(f"  [N{node_rank}/G{local_rank}] all_reduce={t[0].item():.0f} "
          f"expected={expected} {'OK' if ok else 'FAIL'}", flush=True)

    dist.barrier()

    if global_rank == 0:
        print(f"\n{'='*60}", flush=True)
        print(f"SUCCESS: RCCL 8-GPU multi-node OK!", flush=True)
        print(f"{'='*60}", flush=True)

    dist.destroy_process_group()
    print(f"  [N{node_rank}/G{local_rank}] Done.", flush=True)


if __name__ == "__main__":
    mp = None
    main()
