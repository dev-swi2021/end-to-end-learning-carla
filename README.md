# End-to-End learning Carla

# STEP 1 : Imitation Learning
CIL 알고리즘 구현하기.  
논문 제목 : [End-to-end Driving via Conditional Imitation Learning]
(https://arxiv.org/pdf/1710.02410.pdf)  
알고리즘 성능 테스트 : Carla 시뮬레이터에서 자체 성능 평가  
정해진 경로를 잘 가는지 확인(속도, 무사고 등등...)

입력 정보 종류 : RGB, Navigation data, vehicle velocity

# STEP 2 : Reinforcement Learning
미정...(주말을 통해서 서버 컴에서 학습)  
사용 예정 알고리즘 : DDPG, PPO 비교

입력 정보 종류 : RGB, Navigation data, vehicle velocity

# STEP 3 : Imitation Learning + Reinforcement Learning
참고 논문 : [CIRL: Controllable Imitative Reinforcement Learning  for Vision-Based Self-Driving](https://arxiv.org/pdf/1807.03776.pdf)  
기존 CIL 논문에서는 학습했던 정보만을 따라하는 것이기 때문에 새로운 환경에서 적응하기 힘듦.  
새로운 환경에서 적응하기 위한 방법으로 Reinforcement Learning을 사용.  
Reinforcement Learning 알고리즘은 DDPG

입력 정보 종류 : RGB, Semantic Segmentation, Navigation data, vehicle velocity

# STEP 4 : Imitation Learning + Reinforcement Learning + Depth Camera
STEP 3 과정에서 발전한 STEP  
STEP 3의 학습 방식과 동일한데 입력 종류에 Depth Camera 정보를 추가  
정확한 성능 비교를 위해 -> (1단계) Depth Camera + Imitation Learning -> (2단계) Depth Camera + Imitation Learning + Reinforcement Learning