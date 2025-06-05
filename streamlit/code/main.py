import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.cross_decomposition import CCA
from scipy.interpolate import make_interp_spline
import pickle
import datetime

# 한글 폰트 설정
plt.rc('font', family='NanumGothic')  # "NanumGothic" 폰트를 사용
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 폰트 깨짐 방지

# 세션 상태를 사용하여 데이터 유지
if 'df' not in st.session_state:
    st.session_state.df = None

# 사이드바 메뉴
st.sidebar.title("이상감지 시스템")
menu = st.sidebar.selectbox("페이지 선택", ["소개","데이터 로드", "수치해석", "데이터 전처리", "모델 예측", "이상감지"])
# CSV 파일을 업로드하는 페이지
if menu == "데이터 로드":
    st.title("데이터 업로드")
    uploaded_file = st.file_uploader("CSV 파일을 선택하세요", type="csv")

    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.write(st.session_state.df.head())  # 파일의 첫 5행을 미리보기로 보여줌

        
        
        
        
# 이미지 경로 설정
soc1_image_path = "data/soc1.png"
soc2_image_path = "data/soc2.png"
bridge_image_path = "data/다리그림.png"

# 소개 페이지
if menu == "소개":
    st.title("대교 건전성 모니터링 시스템 소개")
    
    # 연구 배경
    st.header("메인 이슈")    
    col1, col2 = st.columns(2)
    with col1:
        st.image(soc1_image_path, caption="SOC 인프라 시설물의 노후화 예시", use_column_width=True)
    with col2:
        st.image(soc2_image_path, caption="SOC 인프라 시설물의 과다 하중 예시", use_column_width=True)
        
    with st.expander("자세히 보기"):
        st.markdown("""
        최근 <span style="color:green;">SOC 인프라 시설물의 노후화</span>와 과다 하중으로 인해 안전 관리의 중요성이 더욱 커지고 있습니다. 
        이러한 시설물은 시간이 지남에 따라 구조적 손상이 발생할 수 있으며, 특히 자연재해나 외부 충격에 의해 위험성이 증대됩니다. 
        노후화된 구조물은 예기치 않은 붕괴의 원인이 될 수 있으며, 이는 <span style="color:green;">인명 피해</span>와 막대한 재산 손실로 이어질 수 있습니다.

        특히, 도로와 교량과 같은 핵심 인프라는 수많은 차량과 보행자의 안전을 책임지고 있기 때문에, 그 상태를 <span style="color:green;">지속적으로 감시하는 것</span>이 필수적입니다. 
        이러한 이유로 법적으로 센서의 설치가 요구되며, 이를 통해 실시간으로 구조물의 상태를 모니터링하여 적시에 필요한 유지 관리 작업을 수행해야 합니다.

        그러나 현재 많은 시설에서 이러한 시스템이 도입되지 못하고 있는 이유는 <span style="color:green;">비용 문제</span>와 센서의 내구성, 계측 신뢰성 등 다양한 어려움 때문입니다. 
        이러한 문제를 해결하지 않으면, 건전성을 확보하지 못한 구조물은 점차 위험해지며, 결국 더 큰 손실을 초래할 수 있습니다.

        따라서 우리는 이러한 시설물의 건전도를 실시간으로 모니터링하고 <span style="color:green;">유지 관리</span>할 수 있는 시스템을 개발하는 것이 목표입니다. 
        이 시스템은 조기 경고를 통해 잠재적 위험을 사전에 식별하고, 유지 관리의 효율성을 높이며, 장기적으로는 구조물의 안전성을 보장하는 역할을 합니다.
        """, unsafe_allow_html=True)
    
    st.info("노후화된 SOC 인프라의 안전 관리에서는 정기적인 모니터링과 효과적인 유지 관리가 필수적입니다. 시스템을 통해 잠재적인 위험을 조기에 발견하고, 시설물의 신뢰성을 높일 수 있습니다.")

    st.divider()
    
    # 진도대교 설명
    st.header("진도대교")
    st.markdown("""
    진도대교는 전라남도 해남군과 진도군을 연결하는 사장교로, 양방향 위쪽의 탑에서 내려온 여러 개의 케이블이 다리 상판을 지탱하는 구조의 교량입니다. 
    """)
    st.image(bridge_image_path, caption="진도대교", use_column_width=True)

    st.divider()

    # 센서 종류 설명
    st.subheader("설치된 센서")
    sensor_data = {
        "센서 코드": [
            "ACC", "CAC", "DIS", "FBG", "FBT", "GNS", "EQK", "WGT ~ WVR"
        ],
        "설명": [
           "지진가속도계",
           "케이블가속도계",
           "처짐계",
           "광섬유 변형률계(교량변형률)",
           "광섬유 온도계",
           "변위계",
           "지진가속도계",
           "풍향 풍속계"
        ],
        "개수": [
            "3  ",  # ACC
            "10 ",  # CAC
            "2   ",  # DIS
            "24  ",  # FBG
            "2   ",  # FBT
            "3   ",  # GNS
            "15  ",  # EQK
            "6   "   # WGT ~ WVR
        ]
    }

    sensor_df = pd.DataFrame(sensor_data)
    st.dataframe(sensor_df, width=800)
    with st.expander("자세히 보기"):
        st.markdown("""
            <p><span style="color:green;">광섬유 변형률계 (FBG)</span>는 교량의 주요 구조부에서 발생하는 미세한 변형을 감지하여, 구조적 결함이나 하중 변화와 같은 신호를 포착합니다.</p>

            <p><span style="color:green;">케이블 가속도계 (CAC)</span>는 케이블에서 발생하는 진동을 측정하여 비정상적인 진동 패턴을 감지합니다. 
            이 두 센서를 종속 변수로 설정함으로써, 교량의 안정성을 포괄적으로 모니터링할 수 있으며, 이는 안정성 측면에서 최우선적으로 중요한 역할을 합니다.</p>

            <p><span style="color:green;">WBT</span> 및 <span style="color:green;">EQK</span> 센서를 통해 감지되는 온도, 바람, 지진 등 
            다양한 불가항력적인 요인들은 교량의 안전에 영향을 미치는 복잡한 외부 요인입니다. 
            이러한 외부 요인들을 독립 변수로 설정함으로써, 정량적 분석을 통해 교량의 안정성을 다각적으로 평가할 수 있습니다.</p>
        """, unsafe_allow_html=True)

    st.info("""
    이를 통해 필요 시 적절한 유지 관리 조치를 신속하게 취할 수 있어, 
    사고를 예방하고 구조물의 수명을 연장하는 데 중요한 역할을 합니다.
    """)
    
    # 강조하고 싶은 부분
    st.header("... 🎯")
    st.markdown("""
    - **실시간 모니터링**: 센서를 통해 수집된 데이터를 바탕으로 대교의 상태를 실시간으로 모니터링합니다.
    - **분석 및 예측**: 수집된 데이터를 분석하여 구조물의 안전 상태를 예측합니다.
    - **유지 관리**: 이상 상황을 빠르게 감지하여 유지보수 비용을 절감합니다.
    """)
    
    st.success("건전성 모니터링 시스템을 통해 대교의 안전을 더욱 강화하고, 유지 보수 비용을 절감할 수 있습니다.")

    


    



# 수치해석 페이지
if menu == "수치해석":
    st.title("수치해석")

    if st.session_state.df is None:
        st.error("데이터가 로드되지 않았습니다.")
    else:
        df = st.session_state.df

        # Unnamed: 0 열 제외
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)

        st.subheader("결측값 처리")

        # 0값이 포함된 열 확인
        zero_columns = df.columns[(df == 0).any()].tolist()
        missing_values = df.isnull().sum().sum()

        if st.button("결측값 확인"):
            st.write("결측값이 포함된 열:", zero_columns)
            st.write(f"총 결측치: {missing_values}개")

        st.subheader("기초 시각화 및 통계량")
        columns_to_visualize = st.multiselect("시각화할 열 선택", df.columns)

        if st.button("시각화 및 통계량 보기"):
            if columns_to_visualize:
                for column in columns_to_visualize:
                    # 기초 통계량
                    st.write(f"{column} 기초 통계량:")
                    st.write(df[column].describe())

                    # 시각화
                    plt.figure(figsize=(10, 5))
                    plt.plot(df[column])
                    plt.title(f"{column} 시각화")
                    plt.xlabel("Index")
                    plt.ylabel("Value")
                    st.pyplot(plt)

                    # 시각화 결과를 세션에 저장
                    st.session_state.data_visualization_plot = plt.gcf()
            else:
                st.warning("시각화할 열을 선택해주세요.")


        st.subheader("상관계수 시각화")
        if st.button("상관계수 시각화"):
            # 수치형 데이터만 선택하여 상관계수 계산
            numeric_df = df.select_dtypes(include=[np.number])  # 수치형 데이터만 선택
            correlation_matrix = numeric_df.corr()

            # 상관계수 히트맵 시각화
            plt.figure(figsize=(12, 8))
            sns.heatmap(correlation_matrix, annot=False, cmap='RdBu', square=True,
                        cbar_kws={"shrink": .8}, linewidths=0.5, vmin=-1, vmax=1)
            st.pyplot(plt)

        # X와 Y 변수 선택
        independent_variables_cac = st.multiselect("독립 변수 선택", df.columns)
        dependent_variables_cac = st.multiselect("종속 변수 선택", df.columns)

        scaling_option = st.selectbox("스케일링 방법 선택", ["MinMaxScaler", "StandardScaler", "Normalizer"])

        st.subheader("정준상관 분석")
        if st.button("정준상관 분석 실행"):
            if independent_variables_cac and dependent_variables_cac:
                # 스케일링
                if scaling_option == "MinMaxScaler":
                    scaler = MinMaxScaler()
                elif scaling_option == "StandardScaler":
                    scaler = StandardScalers()
                else:
                    scaler = Normalizer()

                X_scaled_cac = scaler.fit_transform(df[independent_variables_cac])
                Y_scaled_cac = scaler.fit_transform(df[dependent_variables_cac])

                # 정준상관분석 (CCA)
                cca_cac = CCA(n_components=2)
                X_c_cac, Y_c_cac = cca_cac.fit_transform(X_scaled_cac, Y_scaled_cac)

                # 상관계수 계산
                correlations_cac = np.corrcoef(X_c_cac.T, Y_c_cac.T).diagonal(offset=2)

                # CAC 변수 시각화
                plt.figure(figsize=(15, 7))
                sns.scatterplot(x=X_c_cac[:, 0], y=Y_c_cac[:, 0], hue=df[dependent_variables_cac].mean(axis=1), palette='viridis')
                plt.title(f'CAC (Canonical Correlation: {correlations_cac[0]:.2f})')
                plt.xlabel('Canonical Variable 1 (X)')
                plt.ylabel('Canonical Variable 1 (Y)')
                st.pyplot(plt)

                # 시각화 결과를 세션에 저장
                st.session_state.cca_plot = plt.gcf()

# 데이터 전처리 페이지
if menu == "데이터 전처리":
    st.title("데이터 전처리")

    if st.session_state.df is None:
        st.error("데이터가 로드되지 않았습니다.")
    else:
        df = st.session_state.df

        st.subheader("결측치 처리")
        missing_values = df.isnull().sum().sum()
        st.write(f"총 결측치: {missing_values}개")
        if st.button("결측치 제거"):
            df.dropna(inplace=True)
            st.success("결측치가 제거되었습니다.")

        st.subheader("결측치가 있는 열 처리")
        zero_columns = df.columns[(df == 0).any()].tolist()
        st.write(f"결측치가 포함된 열: {zero_columns}")
        if st.button("결측치가 포함된 열 제거"):
            df = df.drop(columns=zero_columns)
            st.success("결측치가 포함된 열이 제거되었습니다.")

        st.subheader("중복 데이터 처리")
        duplicate_count = df.duplicated().sum()
        st.write(f"중복 데이터 수: {duplicate_count}개")
        if st.button("중복 데이터 제거"):
            df.drop_duplicates(inplace=True)
            st.success("중복 데이터가 제거되었습니다.")

        st.subheader("IQR 이상치 제거")
        if st.button("이상치 제거"):
            numeric_df = df.select_dtypes(include=[np.number])  # 수치형 데이터만 선택
            Q1 = numeric_df.quantile(0.25)
            Q3 = numeric_df.quantile(0.75)
            IQR = Q3 - Q1

            # 이상치 제거
            df = df[~((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)]
            st.success("이상치가 제거되었습니다.")

        # 인덱스 정리
        df.reset_index(drop=True, inplace=True)

        # Unnamed: 0 열 제거
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
            st.success("Unnamed: 0 열이 제거되었습니다.")

        # 최종 전처리된 데이터 보여주기
        st.subheader("최종 전처리된 데이터")
        st.write(df)
        st.session_state.df = df

        # 데이터 저장
        st.subheader("데이터 저장")

        # 데이터 저장
        if st.button("데이터 저장"):
            with open("cleaned_data.pkl", "wb") as f:
                pickle.dump(df, f)
            st.success("데이터가 성공적으로 저장되었습니다.")

        # 다운로드 버튼
        st.download_button("데이터 다운로드", data=pickle.dumps(df), file_name="cleaned_data.pkl", mime="application/octet-stream")


# 모델 예측 페이지
if menu == "모델 예측":
    st.title("모델 예측")

    # 전처리된 데이터 사용
    df = st.session_state.df  # 업데이트된 데이터프레임 참조

    if df is None:
        st.error("데이터가 로드되지 않았습니다.")
    else:
        X_columns = st.multiselect("X 변수 선택", df.columns)
        Y_column = st.selectbox("Y 변수 선택", df.columns)

        if X_columns and Y_column:
            X = df[X_columns]
            y = df[Y_column]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)

            st.write(f"R2 Score: {r2:.4f}")

            # 실제값 vs 예측값 그래프
            plt.figure(figsize=(10, 5))
            plt.plot(y_test.values, label='Actual')
            plt.plot(y_pred, label='Predicted', linestyle='--')
            plt.title('Actual vs Predicted')
            plt.legend()
            st.pyplot(plt)

            # 예측 결과 시각화를 세션에 저장
            st.session_state.model_prediction_results = plt.gcf()

# 이상감지 페이지
if menu == "이상감지":
    st.title("이상감지")

    df = st.session_state.df  # 업데이트된 데이터프레임 참조

    if df is None:
        st.error("데이터가 로드되지 않았습니다.")
    else:
        # 사용자에게 X와 Y 변수를 선택하도록 함
        Y_columns = st.multiselect("Y 변수 선택 (종속변수)", df.columns)
        X_columns = st.multiselect("X 변수 선택 (독립변수)", df.columns)

        threshold = st.slider("Threshold 값 선택", min_value=1, max_value=10, value=4)

        # 이상감지 실행 버튼
        if st.button("이상감지 실행"):
            if X_columns and Y_columns:
                # 선형 회귀 모델 훈련 및 예측
                model = LinearRegression()
                model.fit(df[X_columns], df[Y_columns])
                predictions = model.predict(df[X_columns])

                # 잔차 계산
                residuals = df[Y_columns].values - predictions

                # PF (Probability Filter) 적용
                def probability_filter(residuals, threshold=5):
                    mean_residuals = np.mean(residuals, axis=0)
                    std_residuals = np.std(residuals, axis=0)
                    z_scores = (np.abs(residuals) - mean_residuals) / std_residuals
                    return z_scores > threshold

                # PF 적용하여 이상치 탐지
                pf_anomalies = probability_filter(residuals, threshold)

                # 이상치가 있는 행과 해당 Y 변수 확인
                anomaly_info = []
                for idx in range(pf_anomalies.shape[0]):
                    if any(pf_anomalies[idx]):
                        anomaly_columns = [Y_columns[j] for j in range(len(Y_columns)) if pf_anomalies[idx, j]]
                        anomaly_info.append((df.index[idx], anomaly_columns))

                # 이상치 감지 결과 출력
                st.write(f"탐지된 이상치 개수: {len(anomaly_info)}")
                for idx, cols in anomaly_info:
                    st.write(f"알람: 인덱스 {idx}에 {cols} 변수에서 이상 감지됨.")

                # 그래프 시각화
                for i, col in enumerate(Y_columns):
                    fig, ax = plt.subplots(figsize=(14, 7))

                    # 잔차 보간 (부드러운 선을 위한 보간)
                    index_numeric = np.arange(len(df))
                    y = residuals[:, i]

                    # Cubic spline 보간
                    spline = make_interp_spline(index_numeric, y, k=3)
                    index_smooth = np.linspace(index_numeric.min(), index_numeric.max(), 500)
                    residuals_smooth = spline(index_smooth)

                    # 잔차 및 보간된 선 그래프 그리기
                    ax.plot(index_numeric, residuals[:, i], 'b-', alpha=0.5, label=f'{col} 잔차')
                    ax.plot(index_smooth, residuals_smooth, 'b-', linewidth=2, label=f'{col} 보간 잔차')

                    # PF 이상치 시각화 (네모 박스 사용)
                    pf_anomalies_col = pf_anomalies[:, i]
                    ax.scatter(df.index[pf_anomalies_col], residuals[pf_anomalies_col, i], color='red', marker='s',
                               s=100, edgecolor='black', label=f'{col} 이상치')

                    # 평균 및 표준편차로 임계선 표시
                    ax.axhline(y=np.mean(residuals[:, i]) + 3 * np.std(residuals[:, i]), color='r', linestyle='--',
                               label=f'{col} 상위 임계선')
                    ax.axhline(y=np.mean(residuals[:, i]) - 3 * np.std(residuals[:, i]), color='r', linestyle='--',
                               label=f'{col} 하위 임계선')

                    # 축 범위 설정
                    ax.set_ylim([min(residuals[:, i]) - 1, max(residuals[:, i]) + 1])

                    ax.set_xlabel('Index')
                    ax.set_ylabel('Residuals')
                    ax.set_title(f'{col} 잔차 및 이상치 (PF)')
                    ax.legend()
                    ax.grid(True)

                    # 레이아웃 조정
                    plt.tight_layout()

                    st.pyplot(fig)

                    # 이상 감지 결과 시각화를 세션에 저장
                    st.session_state.anomaly_detection_results = fig
                    
                    
                    
                 