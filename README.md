# image-privacy
1. 运行镜像

   ```cmd
   $docker run -i -p 5000:5000 carc2.easefungame.com:50000/task4:0823
   
   weight /code/vision/data/cbp_trained.pth loaded.
    * Serving Flask app "app" (lazy loading)
    * Environment: production
      WARNING: This is a development server. Do not use it in a production deployment.
      Use a production WSGI server instead.
    * Debug mode: on
    * Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)
    * Restarting with stat
    * Debugger is active!
    * Debugger PIN: 132-041-190
   ```

2. 本地浏览器地址：localhost:5000

3. 上传/test_image文件夹内的测试图像

4. 输出结果显示在网页，同时保存于static文件夹
