import json

def TrainValidTest_split(path,train_path,test_path,valid_path):
    with open(path,"r") as f:
        dataset = json.load(f)
    train_output=[]
    test_output=[]
    valid_output=[]
    for i in range(len(dataset)):
        data = dataset[i]
        split = data["split"]
        if split == "train":
            train_output.append(data)
        elif split =="test":
            test_output.append(data)
        else:
            valid_output.append(data)
        if (i+1) % 100 == 0:
            print("{}/{} completed".format(i+1,len(dataset)))
    print("The train_set capacity:{}".format(len(train_output)))
    print("The test_set capacity:{}".format(len(test_output)))
    print("The valid_set capacity:{}".format(len(valid_output)))
    with open(train_path,"w") as f :
        json.dump(train_output,f,indent=4)
    with open(test_path,"w") as f :
        json.dump(test_output,f,indent=4)
    with open(valid_path,"w") as f :
        json.dump(valid_output,f,indent=4)

if __name__ =="__main__":
    train_path = "data/CUHK-PEDES/CUHK-PEDES-train.json"
    test_path = "data/CUHK-PEDES/CUHK-PEDES-test.json"
    valid_path = "data/CUHK-PEDES/CUHK-PEDES-valid.json"
    dataset_path = "data/CUHK-PEDES/reid_raw.json"
    TrainValidTest_split(dataset_path, train_path, test_path,valid_path)

    train_path = "data/ICFG-PEDES/ICFG-PEDES-train.json"
    test_path = "data/ICFG-PEDES/ICFG-PEDES-test.json"
    valid_path = "data/ICFG-PEDES/ICFG-PEDES-valid.json"
    dataset_path = "data/ICFG-PEDES/ICFG_PEDES.json"
    TrainValidTest_split(dataset_path, train_path, test_path, valid_path)