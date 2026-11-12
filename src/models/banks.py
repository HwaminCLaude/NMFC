"""v3 상태 클래스 — 클래스균형 온라인 FIFO 뱅크와 k-center 코어셋.

설계 규칙: 상태 보관은 src/models에. 삽입 텐서는 전부 detach(requires_grad=False)라
그래디언트는 질의 임베딩으로만 흐른다(정지 경사). LOO는 sample-index 태그로 시행한다.
"""

import torch


class OnlineClassBank:
    """클래스별 링버퍼(FIFO) 교차배치 메모리 뱅크(XBM, Wang et al. 2020).

    z_buf (K,cap,dim)·idx_buf (K,cap) sample-index 태그·valid (K,cap) 유효 마스크·
    count/ptr (K,). t_c에 charge로 일괄 충전하고 매 iter push로 갱신한다.
    """

    def __init__(self, n_classes: int, capacity: int, dim: int, device):
        self.n_classes = n_classes
        self.capacity = capacity
        self.dim = dim
        self.device = device
        self.z_buf = torch.zeros(n_classes, capacity, dim, device=device)
        self.idx_buf = torch.full((n_classes, capacity), -1, dtype=torch.long, device=device)
        self.valid = torch.zeros(n_classes, capacity, dtype=torch.bool, device=device)
        self.count = torch.zeros(n_classes, dtype=torch.long, device=device)
        self.ptr = torch.zeros(n_classes, dtype=torch.long, device=device)

    @torch.no_grad()
    def charge(self, z: torch.Tensor, y: torch.Tensor, idx: torch.Tensor) -> None:
        """t_c 일괄 충전: 클래스별 최신 capacity개로 채운다(z.detach())."""
        z, y, idx = z.detach().to(self.device), y.to(self.device), idx.to(self.device)
        self.z_buf.zero_()
        self.idx_buf.fill_(-1)
        self.valid.zero_()
        self.count.zero_()
        self.ptr.zero_()
        for c in range(self.n_classes):
            mask = y == c
            zc, ic = z[mask], idx[mask]
            n = min(len(zc), self.capacity)
            if n == 0:
                continue
            self.z_buf[c, :n] = zc[-n:]  # 최신 n개
            self.idx_buf[c, :n] = ic[-n:]
            self.valid[c, :n] = True
            self.count[c] = n
            self.ptr[c] = n % self.capacity

    @torch.no_grad()
    def push(self, z: torch.Tensor, y: torch.Tensor, idx: torch.Tensor) -> None:
        """매 iter 갱신: 같은 sample-index가 있으면 그 슬롯 교체(중복 제거), 없으면 FIFO(ptr) 삽입."""
        z, y, idx = z.detach().to(self.device), y.to(self.device), idx.to(self.device)
        for i in range(len(z)):
            c, si = int(y[i]), int(idx[i])
            hit = ((self.idx_buf[c] == si) & self.valid[c]).nonzero(as_tuple=True)[0]
            if len(hit) > 0:
                slot = int(hit[0])
            else:
                slot = int(self.ptr[c])
                self.ptr[c] = (self.ptr[c] + 1) % self.capacity
                if self.count[c] < self.capacity:
                    self.count[c] += 1
            self.z_buf[c, slot] = z[i]
            self.idx_buf[c, slot] = si
            self.valid[c, slot] = True

    @torch.no_grad()
    def reference(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """유효 슬롯을 클래스 순서로 평탄화 → (z_ref (T,dim), y_ref (T,), idx_ref (T,))."""
        rows, ys, idxs = [], [], []
        for c in range(self.n_classes):
            v = self.valid[c]
            if bool(v.any()):
                rows.append(self.z_buf[c][v])
                ys.append(torch.full((int(v.sum()),), c, dtype=torch.long, device=self.device))
                idxs.append(self.idx_buf[c][v])
        return torch.cat(rows), torch.cat(ys), torch.cat(idxs)


def kcenter_greedy(z: torch.Tensor, n_c: int, seed: int = 42) -> torch.Tensor:
    """결정적 farthest-point(k-center greedy) — 선택 인덱스(길이 min(n_c, len(z))).

    첫 중심은 시드 고정 선택, 이후 각 단계에서 기존 중심 집합까지의 최소거리가 최대인 점을 추가.
    커버리지 지향 압축: 클래스 내 밀도를 보존하지 않는다.
    """
    n = len(z)
    n_c = min(n_c, n)
    if n_c <= 0:
        return z.new_empty(0, dtype=torch.long)
    gen = torch.Generator().manual_seed(seed)
    first = int(torch.randint(0, n, (1,), generator=gen).item())
    selected = [first]
    d2 = torch.cdist(z, z[first : first + 1]).pow(2).squeeze(1)  # (n,) 각 점→중심 최소 제곱거리
    for _ in range(n_c - 1):
        nxt = int(torch.argmax(d2).item())
        selected.append(nxt)
        d2 = torch.minimum(d2, torch.cdist(z, z[nxt : nxt + 1]).pow(2).squeeze(1))
    return torch.tensor(selected, dtype=torch.long, device=z.device)
