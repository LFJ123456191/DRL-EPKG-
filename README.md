1.  Download
Clone this repository and navigate to the directory: https://github.com/LFJ123456191/DRL-EPKG-

2、Build Scenarios
Download & build SMARTS according to its repository
[NOTE] The current scenarios are built upon SMARTS v0.4.16, so you may build from source. Ensure that SMARTS is successfully built.


3 Training
We offered the checkpoints with train_logs for all scenarios:


4 Testing
run the scenario test by following example commands:
python test.py DRL-EPKG interaction train_results/Interaction/DRL-EPKG_40/Model/DRL-EPKG_model.h5
