## Factors Influencing Lithography Mask Optimization
#### 참여자 : 14기 여준호(팀장), 13기 백승이, 14기 권나연, 15기 배연욱, 15기 백가은
#### EDA 프로젝트 자료 소개
> * Dataset # 사용한 데이터셋 설명과 URL, 발표자료와 최종 코드 링크
>   * [LithoBench 데이터셋 안내](./dataset/README.md) : 데이터 용량 이슈로 GitHub 직접 업로드 대신 Google Drive 다운로드 링크를 제공합니다.
>   * [SemiConductor_EDA_2602 코드 루트](./code/SemiConductor_EDA_2602/README.md) : 프로젝트 작업 규칙 및 폴더 구조를 확인할 수 있습니다.
> * [EDA 발표자료](Semiconductor Material eda project.pdf) : Factors Influencing Lithography Mask Optimization를 주제로 1. Background, 2. Data Feature Analysis: StdMetal VS StdConact, 3. FFT-Based Efficient Viewing Method, 4. Price Perspective: Via VS Metal의 순서로 발표를 진행하였습니다.
> * [EDA 최종 코드](./code/SemiConductor_EDA_2602/LithoBench) : 팀원별 분석 노트북이 정리된 최종 코드 폴더입니다.

<br>



## EDA 프로젝트 요약

1. 프로젝트 주제 및 목적
        - LithoBench 데이터를 기반으로 리소그래피 공정 단계(target, pixelILT, levelsetILT, litho, resist, printed) 간의 패턴 변화와 오차를 정량화하고 비용(ILT cost) 및 효율 관점에서 주요 영향 요인을 탐색합니다. 추가로 FFT 기반 주파수 영역 분석을 도입하여 패턴 복잡도 및 공간 주파수 특성과 공정 단계별 오차 간의 관계를 정량적으로 탐색하는 것을 목표로 합니다.

2. 데이터 전처리

        - 파일 매칭 점검(누락 파일 확인, CSV 리포트 생성), 그레이스케일 이미지 정규화, Otsu/고정 threshold 기반 이진화, 샘플링(예: 5,000개) 및 단계별 지표 계산을 수행했습니다. 관련 코드: [YeoJunho/pipeline.ipynb](./code/SemiConductor_EDA_2602/LithoBench/YeoJunho/pipeline.ipynb), [BaekGaeun/lithodata (4).ipynb](./code/SemiConductor_EDA_2602/LithoBench/BaekGaeun/lithodata%20(4).ipynb), [PaeYeonUk/2026-01-24_eda_bias-efficiency-cost_v01.ipynb](./code/SemiConductor_EDA_2602/LithoBench/PaeYeonUk/2026-01-24_eda_bias-efficiency-cost_v01.ipynb)
            
 
3. 분석 방법 및 결과
    
        - 분석은 (1) 단계별 IoU/XOR/Boundary 기반 오류 분석, (2) PCA -> UMAP -> KMeans 클러스터링(MetalSet 중심), (3) bias-efficiency-cost 관계 분석 및 그룹별 회귀 시각화로 진행했습니다. 결과적으로 패턴 복잡도/형상 특성에 따라 공정 단계별 오차 분포가 달라졌고, threshold 및 샘플 구성에 따라 민감도 차이가 확인되었습니다. 관련 코드: [KwonNayeon/litho_UMAP.ipynb](./code/SemiConductor_EDA_2602/LithoBench/KwonNayeon/litho_UMAP.ipynb), [PaeYeonUk/2026-01-20_eda_ilt-cost-analysis_v01.ipynb](./code/SemiConductor_EDA_2602/LithoBench/PaeYeonUk/2026-01-20_eda_ilt-cost-analysis_v01.ipynb), [BaekGaeun/lithodata_test.ipynb](./code/SemiConductor_EDA_2602/LithoBench/BaekGaeun/lithodata_test.ipynb)
		    
4. 결론

        - LithoBench 분석 결과, 공정 단계 중 Target → PixelILT 변환 과정에서 오차 기여도가 가장 높게 나타났으며, 초기 ILT 단계에서의 최적화 전략이 전체 printed 성능에 결정적인 영향을 미침을 확인했습니다.
        - 오차와 기하학적 패턴 특성 간 관계를 분석한 결과, Metal 계열은 형상 복잡도와 CD(Critical Dimension)가 복합적으로 작용하는 반면, Via 계열은 Spacing 요소가 지배적인 영향 요인으로 작용함을 확인했습니다. 이는 Layer 특성에 맞춘 차별화된 OPC/ILT 전략 수립이 필요함을 시사합니다.
        - 비용 관점에서는 Via가 Metal 대비 효율성이 낮은 경향을 보였으며 동일한 품질 수준을 달성하기 위해 상대적으로 더 높은 ILT cost가 요구되는 구조적 특성이 관찰되었습니다.
        - 또한 Target 패턴 비교 결과, MetalSet(합성 데이터)이 StdMetal(실제 데이터) 대비 기하학적 복잡도가 높으며 printed 성능이 상대적으로 낮게 나타났습니다. 이러한 복잡성은 공간 주파수 특성으로 해석 가능하며, FFT 기반 분석이 패턴 난이도 및 오차 구조를 효율적으로 파악하는 도구로 활용될 수 있음을 확인했습니다.
    
5. 아쉬운 점
    
        - 대용량 원천 데이터가 저장소에 직접 포함되지 않아 동일한 실험 환경을 완전하게 재현하는 데 한계가 존재합니다. 또한, 반도체 공정 및 소자 특성과 관련된 도메인 지식과 최신 이론을 충분히 반영한 결과 해석과 학술적 검증이 다소 미흡하였습니다. 향후 연구의 신뢰도와 완성도를 제고하기 위해서는 반도체 분야의 전문 지식을 체계적으로 통합하고 정량적 비교군 설정 및 대조 실험을 포함한 추가적인 검증 절차가 필요합니다.

6. 추가로 하면 좋을 분석 방법
    
        - 데이터셋별(StdMetal, StdContact, ViaSet, MetalSet) 일반화 성능 비교, threshold 자동 최적화, 단계별 오차 예측 모델링, 그리고 공정 변수와 품질 지표를 결합한 다변량 민감도 분석을 추가로 진행하면 좋습니다.

<br>



## 각 팀원의 역할
 
|이름|활동 내용| 
|:---:|:---|
|여준호| - (팀장) 전체 일정 관리 및 회의 진행 및 데이터셋 구조/파일 매칭 점검 파이프라인 정리
|백승이| - FFT를 활용한 Analysis of Spectral Entropy 정량화 및 분석
|권나연| - PCA/UMAP/KMeans 기반 클러스터링 분석, 군집 기반 해석 정리
|배연욱| - ILT cost 및 bias-efficiency-cost 분석 및 결과 시각화 및 해석
|백가은| - 단계별 오류 지표(IoU/XOR/Boundary) 분석 및 feature 추출
<br/>



## tree (가능하다면, GitHub에서 tree 명령어를 통해 전체 파일 트리를 보여주세요)
```bash
├── code
│   └── SemiConductor_EDA_2602
│       ├── README.md
│       └── LithoBench
│           ├── BaekGaeun
│           │   ├── README.md
│           │   ├── edaerror.ipynb
│           │   ├── lithodata_test.ipynb
│           │   ├── lithodata (2).ipynb
│           │   └── lithodata (4).ipynb
│           ├── BaekSeungyi
│           │   └── README.md
│           ├── KwonNayeon
│           │   ├── README.md
│           │   ├── litho_UMAP.ipynb
│           │   ├── final_litho_naoniii.ipynb
│           │   ├── final_litho_naoniii_step3X.ipynb
│           │   ├── final_litho_UMAP_fullfit.ipynb
│           │   └── manifest_MetalSet_rep271_PCA_UMAP_KMeans.csv
│           ├── PaeYeonUk
│           │   ├── README.md
│           │   ├── 2026-01-20_eda_ilt-cost-analysis_v01.ipynb
│           │   └── 2026-01-24_eda_bias-efficiency-cost_v01.ipynb
│           └── YeoJunho
│               ├── README.md
│               ├── pipeline.ipynb
│               ├── Newpipe.ipynb
│               ├── edaerror.ipynb
│               ├── lithodata_metal.ipynb
│               ├── eda_6stage_features_errors_StdContact.csv
│               ├── eda_6stage_features_errors_StdContactTest.csv
│               └── eda_6stage_features_errors_StdMetal.csv
├── dataset
│   └── README.md
└── README.md
``` 
