import newton


_, acc, log_loss = newton.run(40, 'Datasets/a9a/train.txt', 'Datasets/a9a/test.txt', type="a9a", method="linesearch_newton", step_size=0.2, H=1)
