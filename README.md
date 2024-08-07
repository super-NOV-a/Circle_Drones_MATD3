# Circle_Drones_MATD3
用于保留我的代码，自定义circle_spread环境下，使用MATD3训练

model文件夹下需要新建circle文件夹

离散动作训练起来依旧困难（状态空间到动作空间的映射太大了），目前简化的内容包括：图像识别—读取黄色像素，动作可以考虑去掉上升下降，无人机们全都保持某一高度。

todo:

图像输入可以再简化为识别的像素输入。并随之修改环境和网络



更新仓库步骤：

```shell
git status
git add .
git commit -m "Some Info"
git push origin main
```

拉取最新仓库步骤：

```shell
git pull origin main
```

