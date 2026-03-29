# Project : 강화학습 공부 프로젝트

## Critical Rules (절대 규칙)
 - 공부용이기 때문에 어느 정도 자세한 주석 필요
 - 어려운 코드보다는 이해하기 쉬운 코드 선호
 - 각 가상환경은 학습, 테스트 코드를 각각 구현해야 하며, 학습 시 best성능을 내는 모델을 저장해야 함
 - 학습 시 best모델이 업데이트 되면 해당 모델이 실시간으로 업데이트 될 수 있도록 테스트코드를 구현해야 함

## Architecture

```
RL_Study/
├── requirements.txt
│
├── study1/                  # CartPole (CPU)
│   ├── train.py             # PPO 학습, EvalCallback으로 best_model 저장
│   ├── test.py              # best_model 실시간 감지 및 테스트
│   ├── models/
│   │   ├── best_model.zip   # EvalCallback이 자동 갱신
│   │   └── final_model.zip  # 학습 완료 후 최종 모델
│   └── logs/                # 평가 로그
│
├── study2/                  # Breakout (GPU)
│   ├── train.py             # CNN+PPO 학습, Atari 전처리 포함
│   ├── test.py              # best_model 실시간 감지 및 테스트
│   ├── models/
│   │   ├── best_model.zip
│   │   └── final_model.zip
│   └── logs/
│
└── study3/                  # Custom Env (GPU)
    ├── custom_env.py        # gymnasium.Env 상속한 커스텀 환경 정의
    ├── train.py             # 커스텀 환경 PPO 학습
    ├── test.py              # best_model 실시간 감지 및 테스트
    ├── models/
    │   ├── best_model.zip
    │   └── final_model.zip
    └── logs/
```


## Library
 - pytorch
 - Stablebaselines3
 - Gymnasium


## Reinforcement Environments
 - study1 : Cartpole (CPU)
 - study2 : Breakout (GPU)
 - study3 : Custom env (GPU)





