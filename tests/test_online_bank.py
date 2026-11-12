import torch

from src.models.banks import OnlineClassBank


def _cpu():
    return torch.device("cpu")


def test_charge_class_balance():
    """충전 후 클래스별 유효 슬롯 = min(n_k, capacity)."""
    bank = OnlineClassBank(n_classes=3, capacity=4, dim=2, device=_cpu())
    z = torch.randn(8, 2)
    y = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1])  # 클래스 0:6, 1:2, 2:0
    bank.charge(z, y, torch.arange(8))
    z_ref, y_ref, _ = bank.reference()
    assert (y_ref == 0).sum() == 4  # min(6,4)
    assert (y_ref == 1).sum() == 2  # min(2,4)
    assert (y_ref == 2).sum() == 0
    assert not z_ref.requires_grad


def test_charge_keeps_latest():
    bank = OnlineClassBank(3, 2, 2, _cpu())
    z = torch.arange(12).float().reshape(6, 2)
    bank.charge(z, torch.zeros(6, dtype=torch.long), torch.arange(6))
    _, _, idx_ref = bank.reference()
    assert set(idx_ref.tolist()) == {4, 5}  # 최신 2개


def test_push_fifo_order():
    bank = OnlineClassBank(1, 3, 2, _cpu())
    for i in range(3):
        bank.push(torch.randn(1, 2), torch.tensor([0]), torch.tensor([i]))
    _, _, idx_ref = bank.reference()
    assert set(idx_ref.tolist()) == {0, 1, 2}
    bank.push(torch.randn(1, 2), torch.tensor([0]), torch.tensor([3]))  # 가장 오래된 0 밀림
    _, _, idx_ref = bank.reference()
    assert set(idx_ref.tolist()) == {1, 2, 3}


def test_push_replaces_same_index():
    bank = OnlineClassBank(1, 3, 2, _cpu())
    bank.push(torch.zeros(1, 2), torch.tensor([0]), torch.tensor([5]))
    bank.push(torch.ones(1, 2), torch.tensor([0]), torch.tensor([5]))  # 같은 idx → 교체
    z_ref, _, idx_ref = bank.reference()
    assert len(idx_ref) == 1
    assert torch.allclose(z_ref[0], torch.ones(2))


def test_insert_tensors_detached():
    bank = OnlineClassBank(1, 3, 2, _cpu())
    z = torch.randn(1, 2, requires_grad=True)
    bank.push(z, torch.tensor([0]), torch.tensor([0]))
    z_ref, _, _ = bank.reference()
    assert not z_ref.requires_grad
