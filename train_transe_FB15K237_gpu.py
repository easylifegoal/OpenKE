import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
with open("./benchmarks/TEST/test2idAll.txt", "r") as f:
	testList = f.readlines()
testTotal = int(testList[0])
print("test total is %s" % testTotal)
count = 0
pipe = 10000
while count < testTotal:
	nextCount = pipe + count
	tmp = []
	if nextCount <= testTotal:
		tmp.append(str(pipe) + "\n")
		tmp.extend(testList[count + 1 : nextCount + 1])
	else:
		tmp.append(str(testTotal - count) + "\n")
		tmp.extend(testList[count + 1 : ])
	with open("./benchmarks/TEST/test2id.txt", "w") as f:
		for item in tmp:
			f.write(item)
	count = nextCount
	# dataloader for training
	train_dataloader = TrainDataLoader(
		in_path = "./benchmarks/TEST/",
		nbatches = 100,
		threads = 8,
		sampling_mode = "normal",
		bern_flag = 1,
		filter_flag = 1,
		neg_ent = 25,
		neg_rel = 0)

	# dataloader for test
	test_dataloader = TestDataLoader("./benchmarks/TEST/", "link")

	# define the model
	transe = TransE(
		ent_tot = train_dataloader.get_ent_tot(),
		rel_tot = train_dataloader.get_rel_tot(),
		dim = 200,
		p_norm = 1,
		norm_flag = True)


	# define the loss function
	# model = NegativeSampling(
	# 	model = transe,
	# 	loss = MarginLoss(margin = 5.0),
	# 	batch_size = train_dataloader.get_batch_size()
	# )

	# train the model
	# trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 1.0, use_gpu = False)
	# trainer.run()
	# transe.save_checkpoint('./checkpoint/transe.ckpt')

	# test the model
	transe.load_checkpoint('./checkpoint/transe.ckpt')
	tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
	tester.run_link_prediction(type_constrain = False)
