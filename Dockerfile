# 基于的基础镜像，无cuda
FROM macio232/torch1.4.0-ubuntu18.04-extras:latest-nocuda

# 维护者信息
MAINTAINER name name@163.com

#用ubuntu国内源替换默认源
RUN rm /etc/apt/sources.list
COPY sources.list /etc/apt/sources.list

# 代码添加到 code 文件夹
ADD ./cbp_cam /code

# 设置 code 文件夹是工作目录
WORKDIR /code

# 安装支持
RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libgl1-mesa-glx libglib2.0-0
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


# 声明镜像内服务监听的端口
EXPOSE 5000

CMD ["python", "/code/app.py"]
