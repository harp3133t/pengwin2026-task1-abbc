# PENGWIN 2026 공식 pelvic/femur 라우터 업데이트 검증

검증일: 2026-07-23  
실행 환경: `conda tabi`, NVIDIA RTX A6000, Docker GPU

## 결론

기존 inference는 조직위의 2026-07-22 업데이트를 정확히 반영하지 않았다.
결정 트리 임계값은 같았지만 SimpleITK와 NumPy 축 대응이 달랐고, 공식 규칙을
주 라우터가 아니라 RF 저신뢰 시 보조 투표로만 사용했다.

현재 구현은 조직위가 게시한 두 함수를 직접 호출하고 그 결과를 최종 case family로
사용하도록 수정됐다.

- `pelvic` → `Sacrum`, `LeftHip`, `RightHip`만 Stage-B 실행
- `femur` → `Femur`만 Stage-B 실행
- legacy 37-feature RF router → Docker 기본값에서 비활성

## 업데이트의 핵심

`SimpleITK.GetSpacing()`은 `(x,y,z)`, `GetArrayFromImage()`는 `(z,y,x)` 순서를
사용한다. 조직위의 구체적인 함수는 아래처럼 배열 축을 기준으로 값을 대응시킨다.

| 공식 변수 | 사용 값 |
|---|---:|
| `spacing_z` | `sp[0]` |
| `spacing_y` | `sp[1]` |
| `spacing_x` | `sp[2]` |
| `physical_z_mm` | `sp[0] * arr.shape[0]` |
| `physical_x_mm` | `sp[2] * arr.shape[2]` |

기존 코드는 `GetSpacing()`을 그대로 `(spacing_x,spacing_y,spacing_z)`로 사용했기
때문에 공식 함수와 다른 feature가 결정 트리에 들어갔다.

## 340-case 전수 재평가

공식 training CT 340개에 공지 함수를 그대로 적용했다.

| GT family | pelvic 예측 | femur 예측 | 정답 |
|---|---:|---:|---:|
| pelvic, 170건 | 139 | 31 | 81.76% |
| femur, 170건 | 14 | 156 | 91.76% |
| 전체 | 153 | 187 | **295/340 = 86.76%** |

축 수정 전 로컬 구현은 172/340 = 50.6%였다. 새 구현은 123건을 추가로 올바르게
라우팅한다.

오분류 case:

- pelvic → femur: 039, 070, 071, 151, 152, 153, 155, 156, 160, 163, 164,
  165, 166, 168, 173, 174, 175, 176, 177, 178, 179, 180, 181, 183, 185,
  187, 188, 189, 191, 193, 195
- femur → pelvic: 254, 278, 283, 297, 303, 335, 342, 346, 359, 377, 389,
  395, 396, 406

조직위 공지는 training set이 100% 일치한다고 말하지 않는다. **Test sets have
been verified to conform to this rule**이라고 명시하므로, training 정확도가 100%가
아닌 것은 공지와 모순되지 않는다.

재현 명령:

```bash
conda run -n tabi python scripts/evaluate_official_case_router.py \
  ../../PENGWIN/PENGWIN26_extracted
```

## 자동 테스트

`tests/test_official_case_router.py`가 다음을 검사한다.

1. 공지의 SimpleITK/NumPy 축 매핑
2. 결정 트리의 모든 분기와 `<=` 경계값
3. 공식 결과가 처리할 anatomy family를 단독으로 선택하는지

결과:

```text
3 passed
```

## Docker 및 GPU smoke test

빌드 이미지:

```text
pengwin-official-router:test
sha256:d841c06d811fd4b566acdeacbd6304f667e0fbce52a00ddf9382597b6c9bbf7a
size: 12,985,592,760 bytes
```

| Case | 공식 route | 출력 label | 실행 시간 | 결과 |
|---|---|---|---:|---|
| 001 | pelvic | 0, 1, 2, 51, 52, 53, 101 | 110.2 s | 성공 |
| 294 | femur | 0, 151, 152, 153, 154 | 67.7 s | 성공 |

두 출력 모두 입력과 size, spacing, origin, direction이 동일했다. 두 실행 모두
`device=cuda`로 Stage-A와 Stage-B checkpoint를 정상 로드했다.

기존 `submission/model_v1_5.tar.gz`는 최상위 archive mode가 700이라 비-root
접근 위험이 확인됐다. 동일 checkpoint를 디렉터리 755/파일 644로 재패키징한
`submission/releases/model_official_router_20260723.tar.gz`를 최종 제출용으로
사용한다.

## 배포 판단

이 변경은 조직위의 명시적인 제출 지침을 따르므로 채택하는 것이 적절하다.
최종 패키징·GPU·무결성 결과는
[`official-case-router-final-results-2026-07-23.md`](official-case-router-final-results-2026-07-23.md)
에 정리했다. Grand Challenge 업로드나 Active 이미지 전환은 인증된 업로드 수단이
없어 수행하지 않았다.
