import torch 
import pickle 
import gzip
import sys
import argparse
import os 

def convert_data(name="dev", cur_path="", target_path="../../../slt/new_data/"):
    trg_size = 151
    skip_frames = 1
    print(name)
    isExist = os.path.exists(target_path)
    print(target_path)
    if not isExist:
    
    # Create a new directory because it does not exist 
        os.makedirs(target_path)
        print("The new directory is created!")

    

    glosses = open(cur_path + '/' + "%s.gloss"%name, "r").readlines()
    textes = open(cur_path + '/'+ "%s.text"%name, "r").readlines()
    names = open(cur_path + '/'+ "%s.files"%name, "r").readlines()
    # write folder.
    write_path = (target_path + "phoenix14t.yang.%s"%name)
    train_write_path = (target_path + "phoenix14t.yang.model.%s"%name)
    file_list = []
    
    file_train_list = []

    if name == "test" or name == "dev":
        output_skeleton_path = "pred_%s_Base_%s.skels"%(name, cur_path.split("/")[1].replace("_", "."))
        if "PT" in cur_path:
            output_skeleton_path = "pred_%s_Base_Gloss2P_counter.skels"%(name)
    else:
        output_skeleton_path = cur_path + '/'+ "%s.skels"%name
        
    train_output_skeleton_path = cur_path + '/'+ "%s.skels"%name

    all_tensors = []
    with open(output_skeleton_path, "r") as f_in:
        lines = f_in.readlines()
        for index in range(len(lines)):

            pruned_tensor = []

            print(glosses[index])
            trg_line = lines[index]
            # convert joints to tensors.
            trg_line = trg_line.split(" ")
            trg_line = [(float(joint) + 1e-8) for joint in trg_line]
            trg_frames = [trg_line[i:i + trg_size] for i in range(0, len(trg_line), trg_size*skip_frames)]

            sign_tensor = torch.Tensor(trg_frames)
            _ , ref_max_idx = torch.max(sign_tensor[:, -1], 0)
            # if ref_max_idx == 0: ref_max_idx += 1
            # Cut down frames by counter
            sign_tensor_cut = sign_tensor[:ref_max_idx+1,:].cpu().numpy()

            trg_frames_cut = [y[:-1] for y in sign_tensor_cut]
            print(len(trg_frames_cut), len(trg_frames))
            # print(glosses[index].strip())
            dict_ = {'name': names[index].strip(),
                    'signer': "Signer08",
                    'gloss': glosses[index].strip(),
                    'text': textes[index].lower().strip(),
                    'sign': torch.Tensor(trg_frames_cut)}
            
            file_list.append(dict_)
    
    # with open(train_output_skeleton_path, "r") as f_in:
    #     lines = f_in.readlines()
    #     for index in range(len(lines)):
    #         print(glosses[index])
    #         trg_line = lines[index]
    #         # convert joints to tensors.
    #         trg_line = trg_line.split(" ")
    #         trg_line = [(float(joint) + 1e-8) for joint in trg_line]
    #         trg_frames = [trg_line[i:i + trg_size] for i in range(0, len(trg_line), trg_size*skip_frames)]
    #         # remove the last counter.
            
    #         trg_frames = [y[:-1] for y in trg_frames]
    #         # print(glosses[index].strip())
    #         dict_ = {'name': names[index].strip(),
    #                 'signer': "Signer08",
    #                 'gloss': glosses[index].strip(),
    #                 'text': textes[index].lower().strip(),
    #                 'sign': torch.Tensor(trg_frames)}
            
    #         file_train_list.append(dict_)
    f = gzip.open(write_path, 'wb')
    pickle.dump(file_list, f)

    # z = gzip.open(train_write_path, 'wb')
    # pickle.dump(file_train_list, z)

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Prepare Data for SLT")
    # parser.add_argument("--name", default="dev", type=str,
    #                         help="Data source on dev/test/train.")
    parser.add_argument("--cur_path", default="Data/train_gloss_between_no_repeat", type=str,
                            help="Data source on dev/test/train.")
                                               
    args = parser.parse_args()

    convert_data('dev', args.cur_path, "../slt/All_data/%s/"%args.cur_path.split("/")[1])
    convert_data("test", args.cur_path, "../slt/All_data/%s/"%args.cur_path.split("/")[1])
    convert_data("train", args.cur_path, "../slt/All_data/%s/"%args.cur_path.split("/")[1])
    # trg_size = 151
    # skip_frames = 1
    # gold = open("test.skels", "r").readlines()
    # with open("../../output_gloss2P.skels", "r") as f_in:
    #     lines = f_in.readlines()
    #     for index in range(len(lines)):
    #         trg_line = lines[index].strip()

    #         gold_line = gold[index].strip()
    #         # convert joints to tensors.
    #         trg_line = trg_line.split(" ")
    #         trg_line = [(float(joint) + 1e-8) for joint in trg_line]
    #         trg_frames = [trg_line[i:i + trg_size] for i in range(0, len(trg_line), trg_size*skip_frames)]

    #         gold_line = gold_line.split(" ")
    #         # print(gold_line)
    #         gold_line = [(float(joint) + 1e-8) for joint in gold_line]
    #         gold_frames = [gold_line[i:i + trg_size] for i in range(0, len(gold_line), trg_size*skip_frames)]
    #         print(torch.Tensor(trg_frames).shape)
    #         # print(gold_frames)
    #         print(torch.Tensor(gold_frames).shape)
    #         print("###")