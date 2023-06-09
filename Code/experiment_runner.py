import newton

#Run Native Newton's Code for ijcnn dataset
newton.run(20, 'Datasets/ijcnn/train', 'Datasets/ijcnn/test', type="ijcnn", method="damped_newton", regularized=False, H=1)

#Run Native Newton's Code for a9a dataset