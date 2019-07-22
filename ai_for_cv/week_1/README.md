# 数据增广

说明: 根据DataAugmentation类创建实例, 传入path(原图路径)和time(增广次数)
效果: 实现随机颜色配合随机变换达到数据增广

1. random_change_color(颜色变换)
操作矩阵, 随机改变每个像素点的颜色值
2. similarity_transform(相似变换, 保留对边平行且保留直角)
cv2.getRotationMatrix2D(1.旋转中心点 2.旋转角度 3.扩缩比例)用于获得模型
cv2.warpAffine(1.原图 2.模型 3.目标图像的长度宽度)用于获得目标图像
3. affine_transform(仿射变换, 保留对边平行, 不保留直角, 需要三个点)
cv2.getAffineTransform(1.对应原图三个点 2.对应目标图像三个点)
cv2.warpAffine(1.原图 2.模型 3.目标图像的长度宽度)用于获得目标图像
4. perspective_transform(投影变换, 不保留对边平行且不保留直角, 需要4个点)
cv2.getPerspectiveTransform(1.对应原图四个点 2.对应目标图像四个点)
cv2.warpPerspective(1.原图 2.模型 3.目标图像的长度宽度)用于获得目标图像
