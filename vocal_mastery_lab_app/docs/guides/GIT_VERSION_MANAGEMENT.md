# Gitバージョン管理ガイド

Vocalis Studioのリリース管理とGitワークフロー。

## 現状（2024-11-27）

| 項目 | 値 |
|-----|-----|
| メインブランチ | `main` |
| 現在のバージョン | `1.0` (Build 1) |
| 既存タグ | なし |

## バージョン番号体系

### セマンティックバージョニング

```
MAJOR.MINOR.PATCH
  │     │     └── バグ修正（後方互換）
  │     └──────── 機能追加（後方互換）
  └────────────── 大きな変更（破壊的変更の可能性）
```

### バージョン例

| バージョン | 内容 |
|-----------|------|
| `1.0.0` | 初回App Storeリリース |
| `1.0.1` | バグ修正のみ |
| `1.1.0` | ピッチ偏差分析機能追加 |
| `2.0.0` | 大規模リニューアル |

### Xcodeでの設定

- **MARKETING_VERSION**: App Storeに表示されるバージョン（例: `1.0.0`）
- **CURRENT_PROJECT_VERSION**: ビルド番号（例: `1`, `2`, `3`...）

同じバージョンで再提出する場合はビルド番号のみインクリメント。

## タグ命名規則

```
v{MAJOR}.{MINOR}.{PATCH}

例:
  v1.0.0  - 初回リリース
  v1.0.1  - Hotfix
  v1.1.0  - 機能追加
```

## ブランチ戦略

### シンプル運用（現在の規模に推奨）

```
main ────●────●────●────●──── (常にリリース可能)
          \              /
           └─ feature/xxx ─┘  (機能開発)
```

- **main**: 常にリリース可能な安定版
- **feature/xxx**: 新機能開発用の一時ブランチ

### ブランチ命名規則

| プレフィックス | 用途 | 例 |
|---------------|------|-----|
| `feature/` | 新機能開発 | `feature/pitch-deviation` |
| `fix/` | バグ修正 | `fix/audio-crash` |
| `docs/` | ドキュメント更新 | `docs/readme-update` |

## リリースワークフロー

### 初回リリース（v1.0.0）

```bash
# 1. 未プッシュのコミットを確認
git status
git log origin/main..HEAD --oneline

# 2. mainにプッシュ
git push origin main

# 3. リリースタグを作成
git tag -a v1.0.0 -m "Initial App Store release - Vocalis Studio 1.0"

# 4. タグをプッシュ
git push origin v1.0.0
```

### 通常リリース（v1.x.x）

```bash
# 1. 機能ブランチで開発
git checkout -b feature/new-feature
# ... 開発 ...
git commit -m "Add new feature"

# 2. mainにマージ
git checkout main
git merge feature/new-feature

# 3. Xcodeでバージョン更新
# MARKETING_VERSION: 1.1.0
# CURRENT_PROJECT_VERSION: インクリメント

# 4. バージョン変更をコミット
git add .
git commit -m "Bump version to 1.1.0"

# 5. タグ作成とプッシュ
git tag -a v1.1.0 -m "Add [feature description]"
git push origin main --tags

# 6. 機能ブランチ削除（オプション）
git branch -d feature/new-feature
```

### Hotfix（緊急バグ修正）

```bash
# 1. mainから直接修正
git checkout main
# ... 修正 ...
git commit -m "Fix critical bug"

# 2. バージョン更新（パッチ番号のみ）
# MARKETING_VERSION: 1.0.1

# 3. タグ作成とプッシュ
git tag -a v1.0.1 -m "Fix [bug description]"
git push origin main --tags
```

## リリースチェックリスト

### リリース前

- [ ] すべてのテストがパス（Unit + UI）
- [ ] 未コミットの変更がない
- [ ] mainブランチにいる
- [ ] Xcodeでバージョン番号を更新済み
- [ ] App Store Connectで新バージョン準備済み

### リリース手順

- [ ] `git push origin main`
- [ ] `git tag -a vX.X.X -m "Release description"`
- [ ] `git push origin vX.X.X`
- [ ] XcodeからArchive → App Store Connectにアップロード
- [ ] TestFlightでテスト
- [ ] App Store審査に提出

### リリース後

- [ ] GitHubでリリースノート作成（オプション）
- [ ] 次バージョンの開発ブランチ作成（必要に応じて）

## タグの確認・管理

```bash
# タグ一覧
git tag -l

# タグの詳細確認
git show v1.0.0

# 特定タグのコードをチェックアウト
git checkout v1.0.0

# タグ削除（ローカル）
git tag -d v1.0.0

# タグ削除（リモート）
git push origin --delete v1.0.0
```

## 過去バージョンへの対応

### 特定バージョンのコード確認

```bash
# タグをチェックアウト（読み取り専用）
git checkout v1.0.0

# 戻る
git checkout main
```

### 過去バージョンからのHotfix（将来的に必要な場合）

```bash
# リリースブランチを作成
git checkout -b release/1.0.x v1.0.0
# ... 修正 ...
git tag -a v1.0.2 -m "Hotfix for 1.0.x"
```

## 推奨事項

1. **タグは必ずリリース時に作成** - App Storeに提出したコードを特定できる
2. **コミットメッセージは明確に** - 何を変更したかわかるように
3. **mainは常にリリース可能に** - 壊れたコードをmainに入れない
4. **機能開発はブランチで** - mainを直接触らない

---

作成日: 2024-11-27
