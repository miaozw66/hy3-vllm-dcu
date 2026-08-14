#!/usr/bin/env python3
"""Single-node RCCL test — run on a single node only. No cross-node needed.

Environment variables:
  RCCL_MASTER_ADDR  — target IP for interface detection (default: 10.16.1.86)
  RCCL_WORLD_SIZE   — number of GPUs to test (default: 8)
"""
import os, sys, socket, fcntl, struct

MASTER_ADDR = os.environ.get("RCCL_MASTER_ADDR", "10.16.1.86")
WORLD_SIZE = int(os.environ.get("RCCL_WORLD_SIZE", "8"))


def main():
    print("[SingleNode] Starting RCCL test...", flush=True)

    # Detect interface
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect((MASTER_ADDR, 1))
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
        iface = "enp45s0f0np0"

    print(f"[SingleNode] iface={iface}, src_ip={src_ip}", flush=True)

    # Env vars
    os.environ["NCCL_SOCKET_IFNAME"] = iface
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["HSA_FORCE_FINE_GRAIN_PCIE"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29599"

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    mp.spawn(worker_fn, nprocs=WORLD_SIZE, join=True)
    print("[SingleNode] ALL DONE!", flush=True)


def worker_fn(local_rank):
    import torch, torch.distributed as dist

    global_rank = local_rank
    world_size = WORLD_SIZE
    print(f"  [G{local_rank}] rank={global_rank} starting...", flush=True)
    torch.cuda.set_device(local_rank)
    gpu = torch.cuda.get_device_name(local_rank)
    print(f"  [G{local_rank}] GPU: {gpu}", flush=True)

    print(f"  [G{local_rank}] Calling init_process_group...", flush=True)
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:29599",
        world_size=world_size,
        rank=global_rank,
    )
    print(f"  [G{local_rank}] RCCL initialized!", flush=True)

    t = torch.ones(1024).cuda() * (global_rank + 1)
    dist.all_reduce(t)
    expected = sum(range(1, world_size + 1))
    ok = abs(t[0].item() - expected) < 0.01
    print(f"  [G{local_rank}] all_reduce={t[0].item():.0f} "
          f"expected={expected} {'OK' if ok else 'FAIL'}", flush=True)

    if global_rank == 0:
        print("SUCCESS: Single-node RCCL works!", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
