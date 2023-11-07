import argparse
import pickle

import matplotlib.pyplot as plt

def plotLosses(total_losses):

    plt.figure(2, figsize=(12, 7))
    plt.clf()
    plt.plot(total_losses, label='Total')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc=1)
    plt.savefig("plot_file.jpg")
    plt.show()

if __name__ == '__main__':

    # style_loss = [
    #     6.6200115, 3.20405, 1.959272,    2.01109, 1.7375725, 1.9026318, 1.12831, 1.199173, 1.4604572, 1.114518, 1.202228,
    #     1.11913254, 1.0885031, 1.06341405, 1.22183122, 1.1838276, 1.6226125, 1.2236056, 1.06364413, 1.04364413
    # ]
    # content_loss = [
    #     2.299655, 1.357877, 1.4855505,  1.380308, 1.215946, 1.18810522, 1.081379, 1.040218, 1.0930986, 1.06199, 1.01919,
    #     0.9854332, 1.0133368, 0.9408155, 0.9730381, 0.94661526, 1.053848, 1.03495912, 0.97154824, 0.98118587
    # ]
    # total_loss = [
    #     8.919667, 4.5619, 3.4448226,  3.3914, 2.953518, 3.09073707, 2.20969, 2.23939, 2.553556, 2.1765, 2.22142,
    #     2.1045658, 2.1018399, 2.00422963, 2.194869, 2.1304428, 2.6764604, 2.258565, 2.0351924, 2.02483
    # ]
    # loading_saved pickle data
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--p', type=str, default=None, help='pickledlosses.pk1')

    args = parser.parse_args()

    if args.p is not None:
        try:
            with open(args.p, 'rb') as file:
                loaded_losses = pickle.load(file)
                (total_loss, epoch) = loaded_losses
                print("Loaded saved losses from file successfully: \n{} \n{}".format(total_loss, epoch))
                plotLosses(total_loss)

        except Exception as e:
            print(f"An error occurred while loading arrays: {str(e)}")
    else:
        pass
    print("Donne")

