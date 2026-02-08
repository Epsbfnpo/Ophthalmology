import os

# === 关键：设置国内镜像 ===

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


from huggingface_hub import login, HfApi


# === 第一步：登录 ===

# 请确保这里填入的是你账号 ZhenlinLiu 下申请的 Write Token



try:

    login(token=my_token, add_to_git_credential=True)

    print(f"登录成功！(当前用户: ZhenlinLiu)")

except Exception as e:

    print(f"登录警告: {e}")


# === 第二步：创建仓库 & 上传 ===

repo_id = "ZhenlinLiu/GDRBench" 

api = HfApi()


print(f"1. 正在检查仓库状态: {repo_id} ...")

try:

    # 尝试创建 (如果已存在会跳过)

    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=True)

    print("   仓库准备就绪！")

except Exception as e:

    print(f"   创建仓库提示: {e}")


print("2. 开始上传数据 (启动大文件专用模式 upload_large_folder)...")

print("   (这可能需要几分钟到几十分钟，支持断点续传，请勿关闭终端)")


try:

    # === 关键修改：使用 upload_large_folder ===

    # 这个函数会自动处理分片、多线程和重试

    api.upload_large_folder(

        folder_path="./GDRBench",

        repo_id=repo_id,

        repo_type="dataset",

        # 默认并发数通常够用，如果报错频繁，可以取消下面这行的注释并改小一点(比如 2 或 4)

        # num_workers=4 

    )

    print("SUCCESS! 所有数据上传完成！")

    print(f"查看地址: https://hf-mirror.com/datasets/{repo_id}")

except Exception as e:

    print(f"上传过程中断: {e}")

    print("提示：请直接重新运行此脚本，它会自动从上次断开的地方继续传。")
