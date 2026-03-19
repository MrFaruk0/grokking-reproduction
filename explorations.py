"""
explorations.py
Further exploration sweep'leri — weight decay, prime p, operations, depth.
Orijinal transformers.py'deki Trainer ve Config üzerine inşa edilmiştir.
"""

import torch
import time
from pathlib import Path
from dataclasses import replace
from transformers import Config, Transformer, Trainer, gen_train_test, full_loss
import helpers


def run_sweep(base_config: Config, overrides: dict, sweep_name: str) -> dict:
    """
    Tek bir sweep run'ı çalıştır.
    overrides: Config field'larını override eden dict (örn. {'weight_decay': 2.0})
    """
    # dataclasses.replace ile yeni config oluştur
    from dataclasses import replace as dc_replace
    cfg = dc_replace(base_config, **overrides)

    model = Transformer(cfg)
    model.to(cfg.device)

    train_data, test_data = gen_train_test(cfg)
    train_tensor = torch.tensor(train_data).to(cfg.device)
    test_tensor  = torch.tensor(test_data).to(cfg.device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr,
        weight_decay=cfg.weight_decay, betas=(0.9, 0.98)
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min(step / 10, 1.0)
    )

    train_losses, test_losses, train_accs, test_accs = [], [], [], []
    grokking_epoch = None

    print(f"\n{'='*50}")
    print(f"Sweep: {sweep_name} | {overrides}")
    print(f"{'='*50}")

    start = time.time()
    for epoch in range(cfg.num_epochs):
        model.train()
        optimizer.zero_grad()

        train_logits = model(train_tensor)[:, -1, :-1]
        train_labels = torch.tensor(
            [cfg.fn(i, j) for i, j, _ in train_data]
        ).to(cfg.device)
        loss = helpers.cross_entropy_high_precision(train_logits, train_labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                test_logits = model(test_tensor)[:, -1, :-1]
                test_labels = torch.tensor(
                    [cfg.fn(i, j) for i, j, _ in test_data]
                ).to(cfg.device)
                test_loss = helpers.cross_entropy_high_precision(
                    test_logits, test_labels
                )

                tr_acc = (train_logits.argmax(-1) == train_labels).float().mean().item()
                te_acc = (test_logits.argmax(-1) == test_labels).float().mean().item()

            train_losses.append(loss.item())
            test_losses.append(test_loss.item())
            train_accs.append(tr_acc)
            test_accs.append(te_acc)

            if te_acc >= 0.95 and grokking_epoch is None:
                grokking_epoch = epoch
                print(f"  → Grokking at epoch {epoch}! test_acc={te_acc:.3f}")

        if epoch % 5000 == 0:
            te = test_losses[-1] if test_losses else float('nan')
            ta = test_accs[-1]   if test_accs   else 0.0
            print(f"  Epoch {epoch:6d} | test_loss={te:.4f} | test_acc={ta:.3f}")

    wall_time = time.time() - start
    epochs_logged = list(range(0, cfg.num_epochs, 100))

    return {
        'config': cfg,
        'overrides': overrides,
        'sweep_name': sweep_name,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'epochs': epochs_logged[:len(train_losses)],
        'grokking_epoch': grokking_epoch,
        'wall_time_min': wall_time / 60,
        'final_test_acc': test_accs[-1] if test_accs else 0.0,
    }


def sweep_weight_decay(base_config: Config,
                       weight_decays=(0.0, 0.1, 0.5, 1.0, 2.0)) -> dict:
    results = {}
    for wd in weight_decays:
        results[wd] = run_sweep(base_config, {'weight_decay': wd},
                                f'weight_decay={wd}')
    return results


def sweep_prime_p(base_config: Config,
                  primes=(53, 97, 113, 127)) -> dict:
    results = {}
    for p in primes:
        from dataclasses import replace as dc_replace
        results[p] = run_sweep(base_config,
                               {'p': p, 'd_vocab': p + 1},
                               f'p={p}')
    return results


def sweep_operations(base_config: Config,
                     operations=('add', 'subtract', 'multiply')) -> dict:
    # 'multiply' orijinal config'de yok, fn_name olarak ekle
    # helpers.py'deki fns_dict'e 'multiply' eklememiz gerekiyor
    results = {}
    for op in operations:
        results[op] = run_sweep(base_config, {'fn_name': op}, f'op={op}')
    return results


def sweep_depth(base_config: Config,
                depths=(1, 2, 3)) -> dict:
    results = {}
    for d in depths:
        results[d] = run_sweep(base_config, {'num_layers': d}, f'depth={d}')
    return results
