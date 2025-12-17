# YUMI-TRACK: Kyudo Form Evaluation System

[![Award](https://img.shields.io/badge/Award-Excellent%20Presentation%20Award-gold)](https://www.sjciee.org/%E5%84%AA%E7%A7%80%E7%99%BA%E8%A1%A8%E8%B3%9E-%E8%8B%B1%E8%AA%9E%E7%99%BA%E8%A1%A8%E5%A5%A8%E5%8A%B1%E8%B3%9E/%E4%BB%A4%E5%92%8C7%E5%B9%B4%E5%BA%A6%E5%8F%97%E8%B3%9E%E8%80%85/)

## 概要 (Overview)

**YUMI-TRACK** は、弓道（Kyudo）のフォームをリアルタイムで解析・評価し、AIによるフィードバックを行うWebアプリケーションです。
MediaPipeを用いた姿勢推定により、「三重十文字（肩・腰・足）」や「等距離（手首・足首）」などの指標を数値化し、理想的な射型とのズレを可視化します。さらに、蓄積されたデータに基づき、OpenAI API (GPT-4o) が改善アドバイスを自動生成します。

本システムは、指導者が不在の環境でも、学生や独習者が自身のフォームを客観的に確認・改善できることを目的としています。

## 受賞・発表 (Awards & Publications)

本研究は以下の学会で高く評価されました。

* **🏆 優秀発表賞 (Excellent Presentation Award)**
  * **大会名**: 令和7年度 電気・電子・情報関係学会四国支部連合大会  
      (2025 Shikoku Section Joint Convention of the Institutes of Electrical and Related Engineers)
  * **発表題目**: リアルタイム姿勢推定による弓道フォーム評価手法の提案とWeb実装
  * **著者**: 奥河 董馬, 益崎 智成, 牧山 隆洋  
      （弓削商船高等専門学校）

## 特徴 (Features)

* **リアルタイム姿勢推定**: Webカメラ映像から骨格を検出し、射法八節に基づいたフォーム評価を即座に行います。
* **ロールベース権限管理**:
  * **Admin**: システム管理者。
  * **Teacher**: 学生管理、全体成績確認、連絡事項の配信。
  * **Practice**: 道場設置の共用端末モード。
  * **Student**: 個人成績の閲覧、生成AIコーチングの受信。
* **AIコーチング**: GPT-4oを活用し、数値データに基づいた具体的な改善アドバイスを提供。
* **データ可視化**: Altairを用いた練習量・スコア推移のグラフ表示。

## 環境構築 (Installation)

動作には **Python 3.x** が必要です。

```bash
pip install -r requirements.txt
```

### OpenAI APIキー設定

プロジェクトのルートディレクトリに `.streamlit` フォルダを作成し、その中に `secrets.toml` ファイルを作成してください。

```toml
[openai]
api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

## 使用方法 (Usage)

```bash
streamlit run kyudo.py
```

## ファイル構成 (File Structure)

```text
.
├── kyudo.py           # メインアプリケーション
├── requirements.txt   # 依存ライブラリ
├── .streamlit/
│   ├── config.toml    # デザイン設定
│   └── secrets.toml   # [必須] OpenAI APIキー設定
├── data1.csv
├── students.csv
├── form.csv
├── data2.csv
├── data2_1.csv
├── data2_2.csv
├── data2_3.csv
├── LICENSE            # MIT License
└── README.md          # 本ドキュメント
```

## Author

奥河 董馬 (Toma Okugawa)  
弓削商船高等専門学校 (National Institute of Technology, Yuge College)
