#from transformers import AutoFeatureExtractor, ResNetForImageClassification
import glob
import os
import torchvision
import torchvision.transforms as transforms
#import timm
import torch
from PIL import Image
#from datasets import load_dataset
import numpy as np
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from cifar10_models.vgg import vgg11_bn
import pickle

adjoint_nrmse = {}
def get_prob_vectors(dataset_path, model, num_classes=10):
    transform = torch.nn.Sequential(
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),)
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    truth_prob_vectors = {}
    recon_prob_vectors = {}
    vector_counts = {}
    reconstruction_nrmse = {}
    count = 0
    vec_sums = torch.zeros(3, device=device)
    for pkl_path in glob.glob(dataset_path, recursive=True):
        with open(pkl_path, 'rb') as f:
            cur_dict = pickle.load(f)
            if 'measurement_var' in cur_dict and abs(cur_dict['measurement_var'] - 0.1) > 0.01:
                continue
            particle_count = float(cur_dict['particle count'])
            pixel_removed_frac = round(float(cur_dict['noise frac']), 1)

            #if abs(float(cur_dict['noise frac']) - pixel_frac_removed) < 0.01 and float(cur_dict['particle count']) == particle_count:
            truth = torch.reshape(cur_dict['truth'].float().to(device), (3, 32, 32))
            recon = torch.reshape(cur_dict['recon'].float().to(device), (3, 32, 32))
            adjoint = torch.reshape(cur_dict['measurement_adjoint'].float().to(device), (3, 32, 32))

            truth_np = (truth * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            Image.fromarray(truth_np, 'RGB').save('cur_true.png')

            adjoint_np = (adjoint * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            Image.fromarray(adjoint_np, 'RGB').save('cur_adjoint.png')

            recon_dir = 'recon/particle_count=' + str(cur_dict['particle count']) + '_pixels_removed=' + str(cur_dict['noise frac'])
            try:
                os.makedirs(recon_dir)
            except OSError as e:
                pass

            cur_counts = 0
            if (pixel_removed_frac, particle_count) in vector_counts:
                cur_counts = vector_counts[(pixel_removed_frac, particle_count)]
            recon_path = f'recon_{cur_counts:06d}.png'
            recon_path = os.path.join(recon_dir, recon_path)
            recon_np = (recon * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            Image.fromarray(recon_np, 'RGB').save(recon_path)

            truth = transform(truth/2 + 0.5)
            recon = transform(recon/2 + 0.5)
            adjoint = transform(adjoint/2 + 0.5)


            vec_sums += torch.mean(truth, dim=(1, 2))
            #print(truth)
            truth_pred = model(truth.unsqueeze(0))
            recon_pred = model(recon.unsqueeze(0))
            truth_ind = int(torch.argmax(truth_pred))
            recon_ind = int(torch.argmax(recon_pred))
            if (pixel_removed_frac, particle_count) not in truth_prob_vectors:
                truth_prob_vectors[(pixel_removed_frac, particle_count)] = np.zeros(num_classes)
                recon_prob_vectors[(pixel_removed_frac, particle_count)] = np.zeros(num_classes)
                vector_counts[(pixel_removed_frac, particle_count)] = 0
                reconstruction_nrmse[(pixel_removed_frac, particle_count)] = 0
                adjoint_nrmse[(pixel_removed_frac, particle_count)] = 0
            if vector_counts[(pixel_removed_frac, particle_count)] > 1000:
                continue
            reconstruction_nrmse[(pixel_removed_frac, particle_count)] += torch.norm(truth - recon)/torch.norm(truth)
            adjoint_nrmse[(pixel_removed_frac, particle_count)] += torch.norm(truth - adjoint)/torch.norm(truth)
            truth_prob_vectors[(pixel_removed_frac, particle_count)][truth_ind] += 1
            recon_prob_vectors[(pixel_removed_frac, particle_count)][recon_ind] += 1
            vector_counts[(pixel_removed_frac, particle_count)] += 1
        count += 1
        if count % 1000 == 0:
            print('it:', count)
            print(truth_prob_vectors)
            print(recon_prob_vectors)



        #img = Image.open(img_path)
        #img = transform(img)
        #pred = model(img.unsqueeze(0))
        #ind = int(torch.argmax(pred))
        ##print(label_img_path)
        ##print('The input picture is classified as [%s].'%
        #  #(class_names[ind]))
        #prob_vectors[ind] += 1
    for key in truth_prob_vectors:
        reconstruction_nrmse[key] /= vector_counts[key]
        truth_prob_vectors[key] /= np.sum(truth_prob_vectors[key])
        recon_prob_vectors[key] /= np.sum(recon_prob_vectors[key])

    return truth_prob_vectors, recon_prob_vectors, reconstruction_nrmse

def compute_cross_entropy(predictions, targets, epsilon=1e-12):
    predictions = torch.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -torch.sum(targets * torch.log(predictions + 1e-9))/N
    return ce

batch_size = 4

#model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
device = torch.device('cuda:2')
model = vgg11_bn(pretrained=True, device=device)
model = model.to(device)
model.eval()
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))])

#print(next(model.parameters()).is_cuda)

#testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                       download=True, transform=transform)
#testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                         shuffle=False, num_workers=2)
#
#classes = ('plane', 'car', 'bird', 'cat',
#           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#
##correct = 0
##total = 0
###since we're not training, we don't need to calculate the gradients for our outputs
#with torch.no_grad():
#    for data in testloader:
#        images, labels = data
#        print(images.shape, images)
#        # calculate outputs by running images through the network
#        outputs = model(images)
#        print('outputs shape:', outputs.shape)
#        #outputs = torch.nn.functional.softmax(outputs, dim=1)
#        # the class with the highest energy is what we choose as prediction
#        _, predicted = torch.max(outputs.data, 1)
#        print('predicted shape:', predicted.shape)
#        total += labels.size(0)
#        correct += (predicted == labels).sum().item()
#
#print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

#pixels_removed_frac = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
truth_path = 'aggregated_truth_probs_same_measurement'
recon_path = 'aggregated_recon_probs_same_measurement'
recon_nrmse_path = 'recon_nrmse_same_measurement'
adjoint_nrmse_path = 'adjoint_nrmse_same_measurement'
#truth_path = 'aggregated_truth_probs'
#recon_path = 'aggregated_recon_probs'
#recon_nrmse_path = 'recon_nrmse'
#adjoint_nrmse_path = 'adjoint_nrmse'
#path = '../experiments_pickle/**/*_pickle'
path = '../same_measurement_experiments_0.9/**/*_pickle'
truth_probs, recon_probs, reconstruction_nrmse = get_prob_vectors(path, model)
#print(truth_probs, recon_probs)
print(adjoint_nrmse)
with open(truth_path, 'bw+') as truth_f:
    pickle.dump(truth_probs, truth_f)
with open(recon_path, 'bw+') as recon_f:
    pickle.dump(recon_probs, recon_f)
with open(recon_nrmse_path, 'bw+') as recon_nrmse_f:
    pickle.dump(reconstruction_nrmse, recon_nrmse_f)
with open(adjoint_nrmse_path, 'bw+') as adjoint_nrmse_f:
    pickle.dump(adjoint_nrmse, adjoint_nrmse_f)

print(recon_probs)

with open(truth_path, 'rb') as truth_f:
    truth_probs = pickle.load(truth_f)
with open(recon_path, 'rb') as recon_f:
    recon_probs = pickle.load(recon_f)
with open(recon_nrmse_path, 'rb') as recon_nrmse_f:
    reconstruction_nrmse = pickle.load(recon_nrmse_f)

prob_l2_error = np.zeros(len(pixels_removed_frac))
recon_nrmse = np.zeros(len(pixels_removed_frac))
confidence_width = np.zeros(len(pixels_removed_frac))
for i, removed_frac in enumerate(pixels_removed_frac):
    cur_tuple = (removed_frac, 5.0)
    prob_l2_error[i] = np.linalg.norm(truth_probs[cur_tuple] - recon_probs[cur_tuple])
    recon_nrmse[i] = reconstruction_nrmse[cur_tuple]
    #confidence_width[i] = np.sum((truth_probs[cur_tuple] - recon_probs[cur_tuple]) ** 4)**(1.0/4.0)/np.sqrt(1000)
    #confidence_width[i] = np.sqrt(2 * 0.1/1000) * 10
    print(removed_frac, truth_probs[cur_tuple], recon_probs[cur_tuple])

plt.plot(pixels_removed_frac, recon_nrmse)
plt.xlabel('Fraction of pixels removed')
plt.ylabel('Average NRMSE')
plt.savefig('nrmse_plot.pdf')
plt.plot(pixels_removed_frac, prob_l2_error)
plt.fill_between(pixels_removed_frac, prob_l2_error-2*confidence_width, prob_l2_error + 2 * confidence_width, alpha =0.1)
plt.savefig('probs.pdf')

exit()
prob_l2_error = np.zeros(12)
label_path = '../experiments/fid-tmp_particle_count=5_noise_frac=0.0/truth/**/*.png'
recon_path = '../experiments/fid-tmp_particle_count=5_noise_frac=0.0/recon/**/*.png'
#label_path = '../fid-tmp/truth/**/*.png'
#recon_path = '../fid-tmp/recon/**/*.png'
label_probs = get_prob_vectors(label_path, model)
recon_probs = get_prob_vectors(recon_path, model)
prob_l2_error[0] = np.linalg.norm(label_probs - recon_probs)

label_path = '../experiments/fid-tmp_particle_count=5_noise_frac=0.05000000074505806/truth/**/*.png'
recon_path = '../experiments/fid-tmp_particle_count=5_noise_frac=0.05000000074505806/recon/**/*.png'
#label_path = '../fid-tmp/truth/**/*.png'
#recon_path = '../fid-tmp/recon/**/*.png'
label_probs = get_prob_vectors(label_path, model)
recon_probs = get_prob_vectors(recon_path, model)
prob_l2_error[1] = np.linalg.norm(label_probs - recon_probs)

label_path = '../experiments/fid-tmp_particle_count=5_noise_frac=0.10000000149011612/truth/**/*.png'
recon_path = '../experiments/fid-tmp_particle_count=5_noise_frac=0.10000000149011612/recon/**/*.png'
#label_path = '../fid-tmp/truth/**/*.png'
#recon_path = '../fid-tmp/recon/**/*.png'
label_probs = get_prob_vectors(label_path, model)
recon_probs = get_prob_vectors(recon_path, model)
prob_l2_error[2] = np.linalg.norm(label_probs - recon_probs)

label_path = '../experiments/fid-tmp_particle_count=5_noise_frac=0.15000000596046448/truth/**/*.png'
recon_path = '../experiments/fid-tmp_particle_count=5_noise_frac=0.15000000596046448/recon/**/*.png'
#label_path = '../fid-tmp/truth/**/*.png'
#recon_path = '../fid-tmp/recon/**/*.png'
label_probs = get_prob_vectors(label_path, model)
recon_probs = get_prob_vectors(recon_path, model)
prob_l2_error[3] = np.linalg.norm(label_probs - recon_probs)

label_path = '../experiments/fid-tmp_particle_count=5_noise_frac=0.20000000298023224/truth/**/*.png'
recon_path = '../experiments/fid-tmp_particle_count=5_noise_frac=0.20000000298023224/recon/**/*.png'
#label_path = '../fid-tmp/truth/**/*.png'
#recon_path = '../fid-tmp/recon/**/*.png'
label_probs = get_prob_vectors(label_path, model)
recon_probs = get_prob_vectors(recon_path, model)
prob_l2_error[4] = np.linalg.norm(label_probs - recon_probs)

label_path = '../experiments/fid-tmp_particle_count=5_noise_frac=0.30000001192092896/truth/**/*.png'
recon_path = '../experiments/fid-tmp_particle_count=5_noise_frac=0.30000001192092896/recon/**/*.png'
#label_path = '../fid-tmp/truth/**/*.png'
#recon_path = '../fid-tmp/recon/**/*.png'
label_probs = get_prob_vectors(label_path, model)
recon_probs = get_prob_vectors(recon_path, model)
prob_l2_error[5] = np.linalg.norm(label_probs - recon_probs)

label_path = '../experiments/fid-tmp_particle_count=5_noise_frac=0.4000000059604645/truth/**/*.png'
recon_path = '../experiments/fid-tmp_particle_count=5_noise_frac=0.4000000059604645/recon/**/*.png'
#label_path = '../fid-tmp/truth/**/*.png'
#recon_path = '../fid-tmp/recon/**/*.png'
label_probs = get_prob_vectors(label_path, model)
recon_probs = get_prob_vectors(recon_path, model)
print(label_probs, recon_probs)
prob_l2_error[6] = np.linalg.norm(label_probs - recon_probs)

label_path = '../experiments/fid-tmp_particle_count=5_noise_frac=0.5/truth/**/*.png'
recon_path = '../experiments/fid-tmp_particle_count=5_noise_frac=0.5/recon/**/*.png'
#label_path = '../fid-tmp/truth/**/*.png'
#recon_path = '../fid-tmp/recon/**/*.png'
label_probs = get_prob_vectors(label_path, model)
recon_probs = get_prob_vectors(recon_path, model)
print(label_probs, recon_probs)
prob_l2_error[7] = np.linalg.norm(label_probs - recon_probs)

label_path = '../experiments/fid-tmp_particle_count=5_noise_frac=0.6000000238418579/truth/**/*.png'
recon_path = '../experiments/fid-tmp_particle_count=5_noise_frac=0.6000000238418579/recon/**/*.png'
#label_path = '../fid-tmp/truth/**/*.png'
#recon_path = '../fid-tmp/recon/**/*.png'
label_probs = get_prob_vectors(label_path, model)
recon_probs = get_prob_vectors(recon_path, model)
print(label_probs, recon_probs)
prob_l2_error[8] = np.linalg.norm(label_probs - recon_probs)

label_path = '../experiments/fid-tmp_particle_count=5_noise_frac=0.699999988079071/truth/**/*.png'
recon_path = '../experiments/fid-tmp_particle_count=5_noise_frac=0.699999988079071/recon/**/*.png'
#label_path = '../fid-tmp/truth/**/*.png'
#recon_path = '../fid-tmp/recon/**/*.png'
label_probs = get_prob_vectors(label_path, model)
recon_probs = get_prob_vectors(recon_path, model)
print(label_probs, recon_probs)
prob_l2_error[9] = np.linalg.norm(label_probs - recon_probs)


label_path = '../experiments/fid-tmp_particle_count=5_noise_frac=0.800000011920929/truth/**/*.png'
recon_path = '../experiments/fid-tmp_particle_count=5_noise_frac=0.800000011920929/recon/**/*.png'
#label_path = '../fid-tmp/truth/**/*.png'
#recon_path = '../fid-tmp/recon/**/*.png'
label_probs = get_prob_vectors(label_path, model)
recon_probs = get_prob_vectors(recon_path, model)
print(label_probs, recon_probs)
prob_l2_error[10] = np.linalg.norm(label_probs - recon_probs)

label_path = '../experiments/fid-tmp_particle_count=5_noise_frac=0.8999999761581421/truth/**/*.png'
recon_path = '../experiments/fid-tmp_particle_count=5_noise_frac=0.8999999761581421/recon/**/*.png'
#label_path = '../fid-tmp/truth/**/*.png'
#recon_path = '../fid-tmp/recon/**/*.png'
label_probs = get_prob_vectors(label_path, model)
recon_probs = get_prob_vectors(recon_path, model)
print('prob_vectors for ', int(pixels_removed_frac[11]), label_probs, recon_probs)
prob_l2_error[11] = np.linalg.norm(label_probs - recon_probs)

print(prob_l2_error)
plt.plot(pixels_removed_frac, prob_l2_error)
plt.xlabel('Fraction of pixels removed')
plt.ylabel('L^2 error')
plt.savefig('plt.pdf')
#transform_fn = transforms.Compose([
#    transforms.Resize(32),
#    transforms.CenterCrop(32),
#    transforms.ToTensor(),
#    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
#])
#net = get_model('cifar_resnet110_v1', classes=10, pretrained=True)
#class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
#               'dog', 'frog', 'horse', 'ship', 'truck']
#
#for label_img_path in glob.glob(label_path, recursive=True):
#    img = Image.open(label_img_path)
#    #img = TF.to_tensor(img)
#    #print(img)
#    #img = torch.reshape(img, (1, 3, 32, 32))
#    img = transform(img)
#    pred = model(img.unsqueeze(0))
#    #pred = model(img)
#    ind = int(torch.argmax(pred))
#    #ind = nd.argmax(pred, axis=1).astype('int')
#    print(label_img_path)
#    print('The input picture is classified as [%s].'%
#      (class_names[ind]))
    


label_dataset = load_dataset('imagefolder', data_dir='fid-tmp/truth')
recon_dataset = load_dataset('imagefolder', data_dir='fid-tmp/recon')
label_dataset = label_dataset['train']['image']
recon_dataset = recon_dataset['train']['image']

#feature_extractor = AutoFeatureExtractor.from_pretrained('microsoft/resnet-18')
#model = ResNetForImageClassification.from_pretrained('microsoft/resnet-18')
#model = timm.create_model('resnet18', num_classes=10, pretrained=True)


label_prob_vectors = get_prob_vectors(label_dataset, model)
recon_prob_vectors = get_prob_vectors(recon_dataset, model)
print(label_prob_vectors)
label_prob_avg = torch.sum(label_prob_vectors, dim=0)/len(label_dataset)
recon_prob_avg = torch.sum(recon_prob_vectors, dim=0)/len(recon_dataset)

print('TV between true and reconstructed:', 0.5 * torch.sum(torch.absolute(label_prob_avg - recon_prob_avg)))

label_combined_class_avg = torch.zeros(2)
label_combined_class_avg[0] = torch.sum(label_prob_avg[:500])
label_combined_class_avg[1] = torch.sum(label_prob_avg[500:1000])

recon_combined_class_avg = torch.zeros(2)
recon_combined_class_avg[0] = torch.sum(recon_prob_avg[:500])
recon_combined_class_avg[1] = torch.sum(recon_prob_avg[500:1000])

print('TV between true and reconstructed (combined classes):', 0.5 * torch.sum(torch.absolute(label_combined_class_avg - recon_combined_class_avg)))


cross_entropy = compute_cross_entropy(recon_prob_vectors, label_prob_vectors)
print(cross_entropy)

cross_entropy = compute_cross_entropy(label_prob_vectors, label_prob_vectors)
print(cross_entropy)
#inputs = feature_extractor(image, return_tensors='pt')

#with torch.no_grad():
    #logits = model(**inputs).logits

#predicted_label = logits.argmax(-1).item()
#print(model.config.id2label[predicted_label])

