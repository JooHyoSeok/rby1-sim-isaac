# RBY1 비주얼 서보잉 pick & place (Isaac Sim)

Rainbow Robotics **RB-Y1**을 NVIDIA Isaac Sim에서 카메라만 보고 상자를 집어서 원하는 위치에 내려놓게 만든 프로젝트야. DGIST *Intelligent Robot Manipulation* 수업의 카메라-로봇 캘리브레이션, 포즈 추정, 비주얼 서보잉 과제를 RBY1 시뮬레이터 위에서 해보려고 [RainbowRobotics/rby1-sim-isaac](https://github.com/RainbowRobotics/rby1-sim-isaac)를 fork해서 만들었어. 시뮬레이터 본체와 SDK 연동은 upstream 그대로고, 이 fork가 얹은 것만 아래에 적었어. (upstream README는 [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md)에 그대로 있어.)

<!-- 데모 영상: GitHub에서 이 README를 편집하면서 mp4 파일을 끌어다 놓으면 이 자리에 재생 링크가 들어가. -->

## 이 fork가 하는 일

- **카메라**: 고정 스테레오 2대와 양손 손등 카메라, 손에는 빨간 구, 상자 3개(파랑·초록·노랑)에는 각자 색 구 landmark를 달았고, 영상과 관절 상태가 시뮬 타임스탬프와 함께 ROS 2 토픽으로 나가.
- **인식**: 스테레오 삼각측량으로 색깔별 상자 포즈를 구해(오차 1~3mm). 코드는 numpy만 써. 쌓은 상자는 맨 위 landmark가 가려져도 3개만 보이면 돼.
- **하이브리드 비주얼 서보잉**: 스테레오로 상자 위치를 대략 잡고, 마지막 정렬은 손등 카메라 영상으로 닫아. 위치 제어와 캘리브레이션이 남기는 정상상태 오차를 영상이 직접 지워.
- **연속 제어**: 20Hz 루프 하나가 목표를 계속 밀어서 SDK 명령 스트림으로 보내. "움직이고 → 멈추고 → 다시 측정"이 없어.
- **전신 IK**: 몸통까지 같이 푸는 Cartesian 명령(rby1 core). 몸통을 접어서 어깨를 낮추고 팔을 굽혀서 일반 높이 테이블에 닿게 했어.
- **충돌 형상**: upstream USD는 손목 링크와 그리퍼에 충돌 형상이 없어서 손이 물체를 통과해. 시작할 때 코드로 손가락에 충돌 형상과 마찰을 붙여(USD는 안 건드려).
- **통합 관제 GUI (Qt)**: 카메라 4개, 위에서 본 장면 지도(상자, 놓을 위치, 테이블을 드래그로 이동), 관절 상태, pick / place / pick and place / home / ABORT.
- **RViz**: URDF(USD에서 추출, 그리퍼 포함)와 TF, 테이블과 상자 마커.

## 실행

필요한 것: NVIDIA GPU와 드라이버, Docker와 nvidia-container-toolkit, 시뮬 이미지 `rainbowroboticsofficial/rby1-sim-isaac:0.10.7-a_v1.2-isaac6.1.0-r1`, **ROS 2 Jazzy 컨테이너**(이름 `intelligent-robotic-manipulation`, `--network=host`, python3와 numpy와 OpenCV 포함: 인식 노드가 여기서 돌아), 호스트의 `/usr/bin/python3`에 PyQt5와 ROS 2 Jazzy(`rclpy`).

```bash
examples/gui/run_gui.sh          # 시뮬, 인식 노드, 실행기, 관제 GUI를 한 번에 (창을 닫으면 전부 정리)
examples/vservo/run_demo.sh      # GUI 없이 한 번: 집어서 들어올리고 결과를 시뮬 정답으로 확인
```

켜는 데 약 2분 걸려(시뮬 부팅). 켠 뒤에는 터미널이나 스크립트, 언어 모델이 같은 실행기를 부를 수 있어:

```bash
examples/gui/rbyctl.py scene                   # 보이는 상자(위치, 어느 상자 위에 있는지), 로봇 상태, 놓을 수 있는 빈 자리
examples/gui/rbyctl.py stack green blue yellow # 아래부터 순서대로, 첫 상자는 제자리
examples/gui/rbyctl.py move blue right         # 파란 상자를 집어서 빈 자리(right | left | near | far | center)에 놓기
examples/gui/rbyctl.py pick yellow             # place, place-on, home, abort ... 은 --help
```

GUI에서는 이렇게 써(GUI의 pick은 파란 상자를 집어):

1. 지도에서 주황 상자나 파란 "place here" 표시를 **끌어서** 옮겨. 팔이 닿을 수 있는 곳(초록)으로 자동으로 붙어. 상자 위에서 휠을 돌리면 방향이 돌아가.
2. `pick`을 누르면 로봇이 준비 자세를 잡은 뒤 서보잉으로 상자를 집어서 들어올려. `place`는 들고 있는 상자를 파란 표시 위치에 내려놓고, `pick and place`는 둘을 이어서 해.
3. `ABORT`로 진행 중인 작업을 멈춰(설계상 동작 중에도 멈추게 만들었는데, 시뮬에서 충분히 검증하진 못했어. 아래 한계 참고).

컨테이너 이름이나 시뮬 이미지가 다르면 `examples/gui/run_gui.sh`와 `examples/vservo/run_demo.sh` 위쪽의 `ROS=`와 `IMAGE=` 값을 고치면 돼.

## 어떻게 동작하나

```
 시뮬 (Isaac Sim, 컨테이너)                        ROS 2 컨테이너                 호스트
 ───────────────────────────                      ──────────────────            ──────────────────
 카메라 4대, 관절 상태, GT  ── ROS 2 토픽 ──────▶  인식 노드                     관제 GUI (Qt)
 장면 명령 서버 (UDP 5110) ◀──────────────────────────────────────────────────── 상자/테이블 이동
 실행기 서비스 (UDP 5120) ◀─────────────── pick / place / abort ────────────────  버튼
   └ 20Hz 서보 루프 → SDK 명령 스트림 → rby1 core (전신 IK) → 로봇
```

- 상자 포즈: 스테레오 두 영상에서 상자 색 구의 중심을 삼각측량하고, 랜드마크를 강체 변환으로 맞춰(Arun SVD). 같은 줄에 놓인 구 두 개는 짝이 두 가지로 맞아서, 짝을 하나로 정하지 않고 강체 배치에 맞는 쪽을 골라.
- 정렬: 손등 카메라가 본 랜드마크로 상자 중심을 손 좌표계에서 구하고, 영상이 찍힌 시점의 손 위치를 써서 지연을 보정한 뒤 도구선 위로 끌어와.
- 자세한 설계 이유는 각 파일 위쪽의 설명(docstring)에 적어놨어.

## 폴더

| 위치 | 내용 |
|---|---|
| `src/rby1_vision.py` | 카메라, landmark, 테이블, 물리 상자, 충돌 형상, ROS 2 발행, 실행 중 장면 명령 (`--vision` 옵션) |
| `examples/gui/` | 관제 GUI, 실행기 서비스(`grasp_core.py`, `servo_server.py`), 명령줄 `rbyctl.py`, 한 줄 실행 `run_gui.sh`, 오프라인 시험 |
| `examples/vservo/` | 인식(`vision.py`), 서보 제어(`servo.py`), 한 번 실행 `run_demo.sh`, 시험 |
| `examples/grasp/` | 서보 없이 위치 제어로 집는 최소 예제 |
| `examples/rviz/` | URDF 추출과 합치기(그리퍼 포함), RViz 설정 |
| `examples/calibration_data/` | 무작위 자세로 이미지와 관절 상태를 모으는 캘리브레이션용 수집 도구 |

## 측정한 결과 (시뮬레이터)

- 스테레오 상자 위치: 1~3mm 오차(랜드마크 3개로 충분).
- 스테레오 추정을 일부러 **(y +2cm, z +5cm) 틀리게** 줬을 때: 위치 제어만 쓰면 손가락이 허공을 잡고 실패하지만, 영상 보정을 켜면 손 좌표계에서 본 오차가 46.9mm에서 0.6mm로 줄고 3.7초 만에 도착해서 집어 올려(손끝 속도는 0에서 7.3cm/s까지 올라갔다 부드럽게 내려와, 중간에 멈추지 않아).
- pick and place: 목표 (0.633, -0.256)에 놓았더니 (0.623, -0.255)에 상자가 똑바로 섰어(x 방향 약 10mm 오차).
- 상자 3개 쌓기(`rbyctl.py stack green blue yellow`, 약 3분): 시뮬 정답으로 초록 z 0.76, 파랑 0.84, 노랑 0.92(8cm씩 정확), 세 상자의 x, y 어긋남은 4mm 이내. 로봇이 스스로 잰 값은 파랑이 초록 위에서 x -9.9mm, 노랑이 파랑 위에서 (-3.2, +2.1)mm. 놓기 직전 정렬이 잰 오차는 2~3mm였는데 놓은 뒤에는 최대 10mm 벌어져서, 놓는 순간 상자가 밀리는지 든 상자 측정에 치우침이 있는지는 아직 몰라.

## 한계와 아직 안 한 것

- **닿는 범위가 좁아**: 팔이 짧아서 기본 테이블에서는 상자가 설 수 있는 위치가 x 0.62~0.65 근처 한 줄뿐이야. 테이블을 로봇 쪽으로 당기면 넓어지지만, 준비 자세로 가기 전에는 테이블 앞 모서리가 x 0.57보다 가까우면 팔꿈치가 걸려(GUI가 막아줘). 그래서 기본 테이블에서는 상자 3개를 12cm 간격 한 줄(x 0.62)로 놓으면 그 띠가 거의 다 차서 `rbyctl.py move`로 옮길 빈 자리가 없고, 쌓기는 되지만 옮기기는 테이블을 당긴 뒤에나 쓸 만해(당긴 테이블은 시뮬에서 검증 전).
- 오른손만 지원하고, 물체는 5x5x8cm 상자 3개야. 그리퍼 방향이 고정이라 상자를 ±30°보다 많이 돌리면 못 집어.
- 시뮬레이터에서만 확인했고 실제 로봇에서는 안 돌려봤어. 영상 지연(0.12초)과 게인은 시뮬에 맞춰 정한 값이야.
- GUI에서 `ABORT`, 테이블을 당긴 뒤 집기, `home`은 아직 시뮬에서 충분히 검증하지 못했어.
- 상자 쌓기는 시뮬에서 한 번 끝까지 성공했을 뿐이야(반복 성공률은 아직 몰라). `move`(빈 자리로 옮기기), 다른 쌓는 순서, 쌓은 상자 다시 풀기, 쌓는 도중 ABORT는 시뮬에서 안 해봤어.
- 카메라-로봇 캘리브레이션(eye-to-hand, hand-eye 풀이)과 ACT 학습은 아직 안 만들었어.

## 시험

시뮬레이터 없이 도는 확인 코드가 있어.

```bash
QT_QPA_PLATFORM=offscreen /usr/bin/python3 examples/gui/test_offline.py   # GUI, 지도, 실행기 명령 처리 (가짜 로봇)
/usr/bin/python3 examples/vservo/test_servo.py                            # 서보 제어식 (핀홀 카메라 모의)
/usr/bin/python3 examples/vservo/test_boxes.py                            # 색깔별 상자 3개 인식, 쌓인 상태 포함 (가짜 스테레오 영상)
```

## 라이선스와 출처

시뮬레이터, USD 에셋, SDK 연동은 Rainbow Robotics의 [rby1-sim-isaac](https://github.com/RainbowRobotics/rby1-sim-isaac)에서 왔어. NVIDIA SPDX 헤더가 있는 파일은 Apache-2.0이야(upstream README 참고). 이 fork가 추가한 코드의 라이선스는 아직 정하지 않았어.
