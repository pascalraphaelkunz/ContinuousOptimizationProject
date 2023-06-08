import native_newton

#Run Native Newton's Code for ijcnn dataset
native_newton.run(10, 'Datasets/ijcnn/train', 'Datasets/ijcnn/test', type="ijcnn")

#Run Native Newton's Code for a9a dataset
native_newton.run(10, 'Datasets/a9a/train.txt', 'Datasets/a9a/test.txt', type="a9a")