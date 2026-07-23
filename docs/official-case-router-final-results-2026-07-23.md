# PENGWIN 2026 공식 case-router 최종 결과표

검증일: 2026-07-23  
환경: `conda tabi`, NVIDIA RTX A6000, Docker `--network none`

## 1. 전체 수행 결과

| 항목 | 결과 | 판정 |
|---|---|---|
| 조직위 공지 원문 확인 | Task 1/2 페이지의 구체적인 `get_image_info()`와 `classify_pelvic_femur()` 확보 | 완료 |
| 공식 축 매핑 반영 | `spacing_z=sp[0]`, `spacing_x=sp[2]` 및 공지 physical FOV 계산 그대로 적용 | 완료 |
| 라우팅 우선순위 변경 | 공식 rule을 최종 family 결정으로 사용 | 완료 |
| legacy RF 비활성 | Docker `PENGWIN_TARGET_ROUTER=0` | 완료 |
| 단위·경계값 테스트 | 3/3 passed | 완료 |
| 340-case 라우터 전수 평가 | 295/340, 86.76% | 완료 |
| Docker 이미지 빌드 | `pengwin-official-router:test` | 완료 |
| offline 검사 | 두 GPU smoke test 모두 `--network none` | 완료 |
| 비-root 실행 | Dockerfile `USER user:user`, 실제 smoke test 성공 | 완료 |
| pelvic GPU smoke | case 001, 110.2초 | 완료 |
| femur GPU smoke | case 294, 67.7초 | 완료 |
| 출력 geometry 검사 | 두 case 모두 size/spacing/origin/direction 일치 | 완료 |
| 알고리즘 `.tar.gz` 생성 | 6.78 GB, gzip 및 `docker load` 통과 | 완료 |
| 기존 모델 `.tar.gz` 감사 | checkpoint는 정상이지만 최상위 권한 `700` 발견 | 사용 금지 |
| 모델 `.tar.gz` 재패키징 | root 755/파일 644, 요구 checkpoint 3개 확인 | 완료 |
| root 추출 → non-root 읽기 | checkpoint 3개 모두 읽기 성공 | 완료 |
| GitHub commit/tag/push | 현재 worktree에 기존 실험 변경이 혼재하고 `gh` 인증 도구 없음 | 보류 |
| Grand Challenge 업로드/Active 전환 | 인증된 GC 업로드 도구·세션 없음 | 보류 |

## 2. 340-case 공식 라우터 결과

| GT family | pelvic 예측 | femur 예측 | 정답률 |
|---|---:|---:|---:|
| pelvic, 170건 | 139 | 31 | 81.76% |
| femur, 170건 | 14 | 156 | 91.76% |
| 전체 | 153 | 187 | **295/340 = 86.76%** |

| 비교 | 정답 | 정확도 | 변화 |
|---|---:|---:|---:|
| 축 수정 전 구현 | 172/340 | 50.59% | 기준 |
| 조직위 업데이트 함수 | 295/340 | 86.76% | **+123건, +36.18%p** |

조직위 보장은 training set 100%가 아니라 **test sets have been verified to
conform to this rule**이다. 따라서 제출 파이프라인은 training 정확도와 관계없이
공식 rule 결과를 우선 사용한다.

## 3. 대표 GPU smoke 결과

| Case | GT family | 공식 route | 처리 anatomy | 출력 라벨 | 시간 |
|---|---|---|---|---|---:|
| 001 | pelvic | pelvic | Sacrum, LeftHip, RightHip | 0, 1, 2, 51, 52, 53, 101 | 110.2 s |
| 294 | femur | femur | Femur | 0, 151, 152, 153, 154 | 67.7 s |

| Case | 기존 전체 FG Dice | 공식 route FG Dice | 변화 |
|---|---:|---:|---:|
| 001 | 0.8897 | **0.9939** | **+0.1042** |
| 294 | 0.9780 | **0.9787** | +0.0007 |

case 001의 큰 개선은 기존 pelvic 출력에 섞였던 femur-range false positive가
공식 route로 제거됐기 때문이다.

## 4. 공식 정렬 proxy 지표

공식 evaluator가 아직 로컬에 제공되지 않아 공개 스펙에 정렬한
`task1_official_aligned_proxy_v2`를 사용했다.

| Case | Fracture Dice | Local Dice | HD95 mm | ASSD mm | Recall | Precision | Instance F1 | Merge | Split | Topology |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 001 | 0.9760 | 0.9760 | 0.878 | 0.148 | 1.000 | 1.000 | 1.000 | 0 | 4 | 0.333 |
| 294 | 0.7995 | 0.7995 | 20.587 | 3.835 | 1.000 | 0.800 | 0.889 | 1 | 6 | 0.250 |

이 proxy는 GT에 존재하는 anatomy 범위만 평가하므로 기존 출력의 wrong-family
false positive는 위 표에서 무시된다. 따라서 이 표가 거의 동일한 것은 정상이며,
라우터의 목적은 hidden test에서 target family 자체를 잘못 선택하는 실패를 막는 것이다.

## 5. 출력 geometry

| Case | Size | Spacing | Origin | Direction | 유효 라벨 범위 |
|---|---|---|---|---|---|
| 001 | 일치 | 일치 | 일치 | 일치 | 0–150, femur 라벨 없음 |
| 294 | 일치 | 일치 | 일치 | 일치 | 0 또는 151–200 |

## 6. 제출 아카이브

| 종류 | 파일 | 크기 | SHA-256 | 검증 |
|---|---|---:|---|---|
| Algorithm container | `submission/releases/pengwin_official_router_20260723.tar.gz` | 6,779,010,693 bytes | `4d5cb4ef1da35a93acf7a112d0ee4decf418d318f0a1b45539c00698e6d2ee31` | `gzip -t` PASS, `docker load` PASS |
| Model, 제출용 | `submission/releases/model_official_router_20260723.tar.gz` | 1,299,497,988 bytes | `433b1d94c51552e723f428c76b20eabca3b5ba134deb863780210f4e37ae9664` | `gzip -t` PASS, root 추출→non-root 읽기 PASS |
| Model, 기존 | `submission/model_v1_5.tar.gz` | 1,299,497,762 bytes | `44d6d94589e5bfda7ec75dc1a8fd02c4bafd35a778317064e258b8e10adbe465` | 최상위 `./` mode 700, **업로드 금지** |

모델 번들에서 확인한 checkpoint:

- Stage A: `Dataset539 ... AnatomyV301 ... fold_0/checkpoint_best.pth`
- Stage B rollback: `Dataset538 ... ABBCPhase1V302 ... fold_0/checkpoint_best.pth`
- Stage B active: `Dataset538 ... AffinityV308 ... fold_0/checkpoint_best.pth`

현재 Docker 설정은 active V308 `fold_0`과 일치하며 legacy RF joblib은 필요하지 않다.

기존 모델 tar의 checkpoint 내용은 정상이지만 최상위 `./`가 `drwx------`여서
root가 그대로 추출하면 비-root 알고리즘 사용자가 모델 경로를 탐색하지 못할 수 있다.
새 제출용 모델 tar는 디렉터리 755, 파일 644, archive owner `0:0`으로 재패키징했다.
Docker root 추출 후 기본 `USER user:user`에서 checkpoint 세 개의 크기와 첫 바이트를
모두 읽어 배포 권한 계약을 검증했다.

## 7. 최종 판단

코드, 라우터 전수 평가, Docker 빌드, non-root/offline GPU 추론, geometry,
알고리즘·모델 아카이브 무결성까지 로컬 제출 준비는 완료됐다.

원격 배포만 남았다. 현재 nested Git 저장소에는 이번 변경 외 기존 affinity 실험
변경이 함께 존재하므로, 변경 범위를 확정하지 않고 commit/tag/push하면 검증 이미지와
다른 release가 될 수 있다. 또한 GitHub CLI와 인증된 Grand Challenge 업로드 세션이
없어 자동 업로드 및 Active 전환은 수행하지 않았다.
