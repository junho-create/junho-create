# CLAUDE.md

이 레포에서 작업을 시작하기 전에 **먼저 [MEMORY.md](MEMORY.md) 를 읽어라.**
프로젝트 목표, 확정된 설계 결정, 원본 데이터 위치, 구현 현황, 다음 할 일이 거기 있다.

한 줄 요약: VL 문서 파싱 데이터셋 큐레이션 엔진(MinerU2.5-Pro 방식). **모델 브링업/DDAS/io/metrics/
normalize/cmcv/cluster/report** 전부 구현+검증됨 (지금 데이터 16886건은 데모용, 실데이터는 나중에
새로 수집 예정). 다음은 cluster→taster CMCV 연결 → hardcase 순. `MEMORY.md` 꼭 먼저 읽을 것.
