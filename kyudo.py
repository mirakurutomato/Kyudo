import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import altair as alt
import os
from datetime import datetime
import csv
from openai import OpenAI
import time


st.set_page_config(
    page_title="YUMI-TRACK",
    page_icon="🏹",
    layout="wide",
    initial_sidebar_state="expanded"
)

frame_placeholder = st.empty()
file_path = 'data1.csv'
file_path2 = 'form.csv'
df1 = pd.read_csv('data1.csv')
df2 = pd.read_csv('students.csv')
df3 = pd.read_csv('data2.csv')
df4 = pd.read_csv('data2_1.csv')
df5 = pd.read_csv('data2_2.csv')
df6 = pd.read_csv('data2_3.csv')
df7 = pd.read_csv('form.csv')

counter = 0


# 関数一覧
class material:
    # キーポイントを取得
    @staticmethod
    def get_keypoint(results, width, height):
        landmarks = results.pose_landmarks.landmark
        # 口の左側のキーポイントを取得
        if landmarks[9].visibility < 0.5:
            return None
        mouth_left_x = int(results.pose_landmarks.landmark[9].x * width)
        mouth_left_y = int(results.pose_landmarks.landmark[9].y * height)
        mouth_left = [mouth_left_x, mouth_left_y]

        # 口の右側のキーポイントを取得
        if landmarks[10].visibility < 0.5:
            return None
        mouth_right_x = int(results.pose_landmarks.landmark[10].x * width)
        mouth_right_y = int(results.pose_landmarks.landmark[10].y * height)
        mouth_right = [mouth_right_x, mouth_right_y]

        # 左肩のキーポイントを取得
        if landmarks[11].visibility < 0.5:
            return None
        left_shoulder_x = int(results.pose_landmarks.landmark[11].x * width)
        left_shoulder_y = int(results.pose_landmarks.landmark[11].y * height)
        left_shoulder = [left_shoulder_x, left_shoulder_y]

        # 右肩のキーポイントを取得
        if landmarks[12].visibility < 0.5:
            return None
        right_shoulder_x = int(results.pose_landmarks.landmark[12].x * width)
        right_shoulder_y = int(results.pose_landmarks.landmark[12].y * height)
        right_shoulder = [right_shoulder_x, right_shoulder_y]

        # 右肘のキーポイントを取得
        if landmarks[14].visibility < 0.5:
            return None
        right_elbow_x = int(results.pose_landmarks.landmark[14].x * width)
        right_elbow_y = int(results.pose_landmarks.landmark[14].y * height)
        right_elbow = [right_elbow_x, right_elbow_y]

        # 右手首(1)のキーポイントを取得
        if landmarks[16].visibility < 0.5:
            return None
        right_wrist_x = int(results.pose_landmarks.landmark[16].x * width)
        right_wrist_y = int(results.pose_landmarks.landmark[16].y * height)
        right_wrist = [right_wrist_x, right_wrist_y]

        # 右手首(2)のキーポイントを取得
        if landmarks[17].visibility < 0.5:
            return None
        right_pinky_x = int(results.pose_landmarks.landmark[17].x * width)
        right_pinky_y = int(results.pose_landmarks.landmark[17].y * height)
        right_pinky = [right_pinky_x, right_pinky_y]

        # 右手首(2)のキーポイントを取得
        if landmarks[17].visibility < 0.5:
            return None
        left_pinky_x = int(results.pose_landmarks.landmark[17].x * width)
        left_pinky_y = int(results.pose_landmarks.landmark[17].y * height)
        left_pinky = [left_pinky_x, left_pinky_y]

        # 右手首(3)のキーポイントを取得
        if landmarks[19].visibility < 0.5:
            return None
        right_index_x = int(results.pose_landmarks.landmark[19].x * width)
        right_index_y = int(results.pose_landmarks.landmark[19].y * height)
        right_index = [right_index_x, right_index_y]

        # 左手首(3)のキーポイントを取得
        if landmarks[19].visibility < 0.5:
            return None
        left_index_x = int(results.pose_landmarks.landmark[19].x * width)
        left_index_y = int(results.pose_landmarks.landmark[19].y * height)
        left_index = [left_index_x, left_index_y]

        # 左股関節のキーポイントを取得
        if landmarks[23].visibility < 0.5:
            return None
        left_hip_x = int(results.pose_landmarks.landmark[23].x * width)
        left_hip_y = int(results.pose_landmarks.landmark[23].y * height)
        left_hip = [left_hip_x, left_hip_y]

        # 右股関節のキーポイントを取得
        if landmarks[24].visibility < 0.5:
            return None
        right_hip_x = int(results.pose_landmarks.landmark[24].x * width)
        right_hip_y = int(results.pose_landmarks.landmark[24].y * height)
        right_hip = [right_hip_x, right_hip_y]

        # 左足首のキーポイントを取得
        if landmarks[27].visibility < 0.5:
            return None
        left_ankle_x = int(results.pose_landmarks.landmark[27].x * width)
        left_ankle_y = int(results.pose_landmarks.landmark[27].y * height)
        left_ankle = [left_ankle_x, left_ankle_y]

        # 右足首のキーポイントを取得
        if landmarks[28].visibility < 0.5:
            return None
        right_ankle_x = int(results.pose_landmarks.landmark[28].x * width)
        right_ankle_y = int(results.pose_landmarks.landmark[28].y * height)
        right_ankle = [right_ankle_x, right_ankle_y]

        return mouth_left, mouth_right, left_shoulder, right_shoulder, right_elbow, right_wrist, right_pinky, left_pinky, right_index, left_index, left_hip, right_hip, left_ankle, right_ankle

    # ベクトルで角度を求める
    @staticmethod
    def angle(ax, ay, bx, by, cx, cy):
        body1 = np.array([ax, ay])
        body2 = np.array([bx, by])
        body3 = np.array([cx, cy])

        a = body1 - body3
        b = body2 - body3

        u = np.linalg.norm(a)
        v = np.linalg.norm(b)

        cos = np.dot(a, b) / (u * v)
        cos = np.clip(cos, -1.0, 1.0)
        true_angle = np.arccos(cos)
        degree = np.degrees(true_angle)

        return degree


def connected(name, series, description):
    with open(file_path2, mode='r', encoding='utf-8') as fn3:
        rows = list(csv.reader(fn3))

        rows.append([name, series, description])

    with open(file_path2, 'w', newline='', encoding='utf-8') as fn4:
        writer = csv.writer(fn4)
        writer.writerows(rows)         
    
    st.success("送信に成功しました！")


def save_to_csv(text, length, body1, body2, body3, total):
    file_path = 'students.csv'
    file_exists = os.path.exists(file_path)

    with open(file_path, mode='a', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f, lineterminator='\r\n')
        if not file_exists:
            writer.writerow(['名前','等距離','三重十文字（肩）','三重十文字（腰）','三重十文字（両足土踏まず）','合計得点'])
        
        results = [
            text,
            max(0, round(100 - (length * 2))),
            max(0, round(100 - abs(body1 * 2))),
            max(0, round(100 - abs(body2 * 2))),
            max(0, round(100 - abs(body3 * 2))),
            max(0, round(total))
        ]
        writer.writerow(results)


# ログイン後の画面表示
def show_main_page():
    st.sidebar.markdown(f"ようこそ！ {st.session_state['username']}さん")

    if st.sidebar.button("ログアウト", key="logout_button"):
        st.session_state.clear()
        st.query_params.clear()
        st.rerun()

    user_type = st.session_state["user_type"]

    if user_type == "admin":
        show_admin_page()
    elif user_type == "teacher":
        show_teacher_page(st.session_state["username"])
    elif user_type == "practice":
        show_practice_page(st.session_state["username"])
    elif user_type == "student":
        show_student_page(st.session_state["username"])


def show_login_page():
    st.title("ログインページ")

    username = st.text_input("ユーザー名")
    password = st.text_input("パスワード", type="password")

    if st.button("ログイン", key="login_button"):
        for user_type, cred_dict in USER_CREDENTIAL_SETS.items():
            if username in cred_dict and cred_dict[username] == password:
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.session_state["user_type"] = user_type

                st.query_params = {"user": username}

                st.success("ログイン成功！")
                st.rerun()
        else:
            st.error("ユーザー名またはパスワードが間違っています。")


def run_camera(text):
    global counter, results1, results2, results3, results4, results5, df2, right_arm_angle, total, frame_placeholder
    while True:
        counter, total = (0, None)
        results1, results2, results3, results4, results5 = ("None", "None", "None", "None", 0)
        hold_start_time = None
        hold_duration_required = 4.0

        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        mesh_drawing_spec = mp_drawing.DrawingSpec(thickness=1, color=(0, 255, 0))
        mark_drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=3, color=(255, 0, 0))
        cap_file = cv2.VideoCapture(0)

        with mp_pose.Pose(min_detection_confidence=0.7, static_image_mode=False) as pose_detection:

            while cap_file.isOpened():
                success, image = cap_file.read()
                if not success:
                    break

                image = cv2.resize(image, dsize=None, fx=1.5, fy=1.5)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                height, width = rgb_image.shape[:2]

                results = pose_detection.process(rgb_image)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image=rgb_image,
                        landmark_list=results.pose_landmarks,
                        connections=mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mark_drawing_spec,
                        connection_drawing_spec=mesh_drawing_spec
                    )

                    keypoints = material.get_keypoint(results, width, height)
                    if keypoints is None:
                        continue
                    (mouth_left, mouth_right, left_shoulder, right_shoulder, right_elbow, right_wrist, right_pinky, left_pinky, right_index, left_index, left_hip, right_hip, left_ankle, right_ankle) = keypoints

                    threshold = 15
                    mouth_y = (mouth_left[1] + mouth_right[1]) / 2
                    hit_kai = abs(right_wrist[1] - mouth_y) < threshold or abs(right_pinky[1] - mouth_y) < threshold or abs(right_index[1] - mouth_y) < threshold

                    condition = hit_kai and right_elbow[0] < right_wrist[0]

                    if condition:
                        if hold_start_time is None:
                            hold_start_time = time.time()
                        elif time.time() - hold_start_time >= hold_duration_required:
                            shoulder = ((right_pinky[0] + right_index[0]) / 2) - ((left_pinky[0] + left_index[0]) / 2)

                            body_s = left_shoulder[1] - right_shoulder[1]
                            body_h = left_hip[1] - right_hip[1]
                            body_a = left_ankle[1] - right_ankle[1]

                            foot = left_ankle[0] - right_ankle[0]
                            length = abs(foot - shoulder)
                            right_arm_angle = material.angle(
                                right_shoulder[0], right_shoulder[1],
                                right_wrist[0], right_wrist[1],
                                right_elbow[0], right_elbow[1]
                            )

                            parallel_s = material.angle(
                                (mouth_left[0] + mouth_right[0]) / 2, (mouth_left[1] + mouth_right[1]) / 2,
                                left_shoulder[0], left_shoulder[1],
                                (left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2
                            )

                            parallel_h = material.angle(
                                (mouth_left[0] + mouth_right[0]) / 2, (mouth_left[1] + mouth_right[1]) / 2,
                                left_hip[0], left_hip[1],
                                (left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2
                            )

                            parallel_a = material.angle(
                                (mouth_left[0] + mouth_right[0]) / 2, (mouth_left[1] + mouth_right[1]) / 2,
                                left_ankle[0], left_ankle[1],
                                (left_ankle[0] + right_ankle[0]) / 2, (left_ankle[1] + right_ankle[1]) / 2
                            )

                            body1 = 90 - parallel_s
                            body2 = 90 - parallel_h
                            body3 = 90 - parallel_a

                            total = (max(0, round(100 - (length * 2)))
                                     + max(0, round(100 - abs(body1 * 2)))
                                     + max(0, round(100 - abs(body2 * 2)))
                                     + max(0, round(100 - abs(body3 * 2)))) / 4
                            break
                    else:
                        hold_start_time = None

                    frame_placeholder.image(rgb_image, caption="リアルタイム映像", use_container_width=True)

        if total is not None:
            img = np.ones((400, 600, 3), dtype=np.uint8) * 255
            cv2.putText(img, "--score--", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0))
            cv2.putText(img, f"equidistance {max(0, round(100 - (length * 2)))}", (20, 100), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0))
            cv2.putText(img, f"body_shoulder {max(0, round(100 - abs(body1 * 2)))}", (20, 150), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0))
            cv2.putText(img, f"body_hip {max(0, round(100 - abs(body2 * 2)))}", (20, 200), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0))
            cv2.putText(img, f"body_ankle {max(0, round(100 - abs(body3 * 2)))}", (20, 250), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0))
            cv2.putText(img, f"total {max(0, round(total))}", (20, 325), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0))

            if not st.button("測定終了"):
                frame_placeholder.image(img, caption="スコア結果", use_container_width=True)
                cv2.waitKey(1)

                st.write("改善すべき箇所・改善方法")

                if foot - shoulder < -5:
                    st.write("・等距離：両手首間の距離の方が、両足首間の距離より長くなっています")
                elif foot - shoulder > 5:
                    st.write("・等距離：両足首間の距離の方が、両手首間の距離より長くなっています")
                    
                if body1 < -5 or body1 > 5:
                    st.write("・三重十文字（肩）：三重十文字（肩）が直角になれていません。肩水平と体の中心軸と直角になるよう、練習を重ねましょう。")

                if body2 < -5 or body2 > 5:
                    st.write("・三重十文字（腰）：三重十文字（腰）が直角になれていません。肩水平と体の中心軸と直角になるよう、練習を重ねましょう。")

                if body_s < -5:
                    st.write("（肩水平：右肩の方が左肩よりも高くなっています）")
                elif body_s > 5:
                    st.write("（肩水平：左肩の方が右肩よりも高くなっています）")

                if body3 < -5 or body3 > 5:
                    st.write("・三重十文字（両足土踏まず）：三重十文字（両足土踏まず）が直角になれていません。肩水平と体の中心軸と直角になるよう、練習を重ねましょう。")
                
                if body_h < -5:
                    st.write("（腰水平：右腰の方が左腰よりも高くなっています）")
                elif body_h > 5:
                    st.write("（腰水平：左腰の方が右腰よりも高くなっています）")

                if body3 < -5 or body3 > 5:
                    st.write("・三重十文字（両足土踏まず）：三重十文字（両足土踏まず）が直角になれていません。肩水平と体の中心軸と直角になるよう、練習を重ねましょう。")

                if body_a < -5:
                    st.write("（両足土踏まず水平：右足土踏まずの方が左足土踏まずよりも高くなっています）")
                elif body_a > 5:
                    st.write("（両足土踏まず水平：左足土踏まずの方が右足土踏まずよりも高くなっています）")
                            
            counter = 1
            save_to_csv(text, length, body1, body2, body3, total)
            cap_file.release()
            cv2.destroyAllWindows()
            break

        else:
            break


def show_admin_page():
    global df4
    st.title("管理者新規登録")
    st.info("名前とパスワード、所属を書いて下さい。")

    keyword = st.text_input("新規登録用名前")
    n_password = st.text_input("新規登録用パスワード", type="password")
    n_belong = st.text_input("新規登録用所属")

    if st.button("登録"):
        if keyword and n_password and n_belong:
            if keyword in df4["名前"].values:
                st.error("この名前はすでに登録されています。")
            else:
                new_data = pd.DataFrame([[keyword, n_password, n_belong]], columns=["名前", "パスワード", "所属"])
                df4 = pd.concat([df4, new_data], ignore_index=True)
                df4.to_csv('data2_1.csv', index=False)
                st.success("管理者を登録しました！")
        else:
            st.warning("全ての項目を入力して下さい。")


def show_teacher_page(username):
    global df4, df5
    pagelist = ["TOP", "email", "practice"]
    
    selector = st.sidebar.selectbox("ページ選択", pagelist)

    if selector == "TOP":
        st.title("測定システム新規登録")
        st.info("名前とパスワードを書いて下さい。")

        keyword = st.text_input("新規登録用名前")
        n_password = st.text_input("新規登録用パスワード", type="password")
        belong_row = df4[df4['名前'] == username]
        if not belong_row.empty:
            belong = belong_row.iloc[0, 2]
        else:
            belong = "所属不明"

        if st.button("登録"):
            if keyword and n_password and belong:
                if keyword in df5["名前"].values:
                    st.error("この名前はすでに登録されています。")
                else:
                    new_data = pd.DataFrame([[keyword, n_password, belong]], columns=["名前", "パスワード", "所属"])
                    df5 = pd.concat([df5, new_data], ignore_index=True)
                    df5.to_csv('data2_2.csv', index=False)
                    st.success("管理者を登録しました！")
            else:
                st.warning("全ての項目を入力して下さい。")
    
    if selector == "email":
        st.title("伝達事項送信欄")
        with st.form("my_form", clear_on_submit=False):
            name = st.text_input('名前を入力して下さい')
            series = st.text_input('タイトル')
            description = st.text_area('説明')

            submitted = st.form_submit_button("送信")
        
            if submitted:
                if name and series and description:
                    connected(name, series, description)
                
                else:
                    st.warning("全ての項目を入力して下さい。")
    
    if selector == "practice":
        data = df1.rename(columns={"日付": "data", "練習回数": "count"})
        data["count"] = pd.to_numeric(data["count"], errors='coerce').fillna(0).astype(int)

        st.header('ALL practice')
        fig = alt.Chart(data).mark_bar().encode(
            x='data:N',
            y='count:Q',
            tooltip=['data', 'count']
            )
        st.altair_chart(fig, use_container_width=True)

        st.header('Grades')
        last_10_rows2 = df2.tail(10).iloc[::-1]
        st.dataframe(last_10_rows2)


def show_practice_page(username):
    pagelist = ["TOP", "new"]

    selector = st.sidebar.selectbox("ページ選択", pagelist)

    if selector == "TOP":
        st.header('フォーム測定')

        st.info('ユーザー名を書いた後、測定ボタンを押して下さい', icon=None)

        text = st.text_input("ユーザー名を書いて下さい")
        if st.button('測定ボタン', key=0):
            if text:
                run_camera(text)

                file_path = 'data1.csv'
                today = datetime.today()
                date_str = f"{today.month}月{today.day}日"

                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.DataFrame(columns=["日付", "練習回数"])

                if date_str in df["日付"].values:
                    df.loc[df["日付"] == date_str, "練習回数"] += counter
                else:
                    df = pd.concat([df, pd.DataFrame({"日付": [date_str], "練習回数": [counter]})], ignore_index=True)

                df.to_csv(file_path, index=False, encoding='utf-8')
        
        data = df1.rename(columns={"日付": "data", "練習回数": "count"})
        data["count"] = pd.to_numeric(data["count"], errors='coerce').fillna(0).astype(int)

        st.header('ALL practice')
        fig = alt.Chart(data).mark_bar().encode(
            x='data:N',
            y='count:Q',
            tooltip=['data', 'count']
            )
        st.altair_chart(fig, use_container_width=True)

        st.header('Grades')
        last_10_rows2 = df2.tail(10).iloc[::-1]
        st.dataframe(last_10_rows2)

    if selector == "new":
        st.title("ユーザ新規登録")
        st.info("名前とパスワードを書いて下さい。")

        df6 = pd.read_csv('data2_3.csv')

        keyword = st.text_input("新規登録用名前")
        n_password = st.text_input("新規登録用パスワード", type="password")

        belong_row = df5[df5['名前'] == username]
        belong = belong_row.iloc[0, 2] if not belong_row.empty else "所属不明"

        if st.button("登録"):
            if keyword and n_password:
                if keyword in df6["名前"].values:
                    st.error("この名前はすでに登録されています。")
                else:
                    new_data = pd.DataFrame([[keyword, n_password, belong]], columns=["名前", "パスワード", "所属"])
                    df6 = pd.concat([df6, new_data], ignore_index=True)
                    df6.to_csv('data2_3.csv', index=False)
                    st.success("ユーザを登録しました！")
            else:
                st.warning("全ての項目を入力して下さい。")

        st.subheader("登録しているユーザ名")
        st.dataframe(df6["名前"])


def show_student_page(username):    
    pagelist = ["成績", "伝達事項"]

    selector=st.sidebar.selectbox("ページ選択",pagelist)
    if selector=="成績":
        st.title(f"{username}の成績")
    
        df_person = df2[df2['名前'] == username]
        df_recent = df_person.tail(5).copy()
        df_recent = df_recent.reset_index(drop=True)
        df_recent = df_recent[::-1].reset_index(drop=True)

        avg_leg = df_person['等距離'].mean()
        avg_body1 = df_person['三重十文字（肩）'].mean()
        avg_body2 = df_person['三重十文字（腰）'].mean()
        avg_body3 = df_person['三重十文字（両足土踏まず）'].mean()

        col1, col2 = st.columns(2)

        with col1:
            st.header("等距離")
            st.bar_chart(df_recent['等距離'])
    
        with col2:
            st.header("三重十文字（肩）")
            st.bar_chart(df_recent['三重十文字（肩）'])

        col3, col4 = st.columns(2)
    
        with col3:
            st.header("三重十文字（腰）")
            st.bar_chart(df_recent['三重十文字（腰）'])
        
        with col4:
            st.header("三重十文字（両足土踏まず）")
            st.bar_chart(df_recent['三重十文字（両足土踏まず）'])
    
        openai_api_key = st.secrets["openai"]["api_key"]

        st.header("生成AIによるアドバイス")

        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "何か気になることはありますか？"}]

            first_user_msg = (
                f"私の弓道の直近の平均成績です。アドバイスを下さい。"
                f"両手首間の距離と両足首間の距離が等しい状態：{avg_leg:.1f}点、三重十文字（肩）：{avg_body1:.1f}点、"
                f"三重十文字（腰）：{avg_body2:.1f}点、三重十文字（両足土踏まず）：{avg_body3:.1f}点"
                )
        
            st.session_state["messages"].append({"role": "user", "content": first_user_msg})
            with st.chat_message("user"):
                st.write(first_user_msg)
            
            client = OpenAI(api_key=openai_api_key)
            response = client.chat.completions.create(
                messages=st.session_state["messages"],
                model="gpt-4o"
            )
            msg = response.choices[0].message
            st.session_state["messages"].append(msg)
            with st.chat_message("assistant"):
                st.write(msg.content)
    
    if selector=="伝達事項":
        st.header("指導者からの伝達事項")
        st.dataframe(df7.iloc[-10:])


def load_credentials(filepath):
    df = pd.read_csv(filepath)
    return dict(zip(df["名前"], df["パスワード"]))

USER_CREDENTIAL_SETS = {
    "admin": load_credentials("data2.csv"),
    "teacher": load_credentials("data2_1.csv"),
    "practice": load_credentials("data2_2.csv"),
    "student": load_credentials("data2_3.csv"),
}

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["username"] = ""
    st.session_state["user_type"] = ""

if "user" in st.query_params and not st.session_state["logged_in"]:
    username = st.query_params["user"]
    for user_type, cred_dict in USER_CREDENTIAL_SETS.items():
        if username in cred_dict:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.session_state["user_type"] = user_type
            break


if st.session_state["logged_in"]:
    show_main_page()
else:

    show_login_page()
