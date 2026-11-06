# TrackingExperience

### コミットメッセージ例
add: 新機能・ファイルの追加
fix: バグや不具合の修正
refactor: コードの可読性向上や構造の整理
docs: ドキュメントの変更
chore: 雑務，設定ファイルの変更

### recordingのディレクトリ構造
recordings/
├── aim_<name><mouseDPI>_<date>.../
│   ├── trial_00_static_pre/
│   ├── trial_01_slow_s1/
│   ├── trial_02_slow_s2/
│   │   ├── joints/
│   │   │   └── rs_joints.npz
│   │   └── task.csv
│   ├── trial_03_slow_s3/
│   ├── trial_04_fast_s1/
│   ├── trial_05_fast_s2/
│   └── trial_06_fast_s3/
├── aim_<name><mouseDPI>_<date>.../
├── aim_<name><mouseDPI>_<date>.../
└── ...