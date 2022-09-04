import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data.sampler import BatchSampler, RandomSampler

import numpy as np

import os
from PIL import Image
from sklearn.datasets import fetch_olivetti_faces



#The dataset for tiny imagenet 
class TinyImagenet(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
    ):
        """TinyImageNet dataset for pytorch.
        You have to prepare the dataset yourself before use.
        
        Args:
            root: directory path for dataset images. (included train and test)
            train: load train data. Defaluts to True.
            transform: torchvision transforms for image. Defaults to None.
            target_transform: torchivison trasnsform for label. Defaults to None.
        """
        root = os.path.expanduser(root)
        
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform
            
        self.target_transform = target_transform
        
        self.train = train
        if self.train:
            self.data = np.load(root + '/train/img_train.npy')
            self.targets = np.load(root + '/train/label_train.npy').astype(np.int64)
            #self.targets = torch.from_numpy(self.targets)
            self.num = len(self.data)
        else:
            self.data = np.load(root + '/test/img_test.npy')
            self.targets = np.load(root + '/test/label_test.npy').astype(np.int64)
            #self.targets = torch.from_numpy(self.targets)
            self.num = len(self.data)
        
        self.data = np.uint8(self.data)
        self.classes = ['goldfish, Carassius auratus', 'European fire salamander, Salamandra salamandra',
                        'bullfrog, Rana catesbeiana', 'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui',
                        'American alligator, Alligator mississipiensis', 'boa constrictor, Constrictor constrictor',
                        'trilobite', 'scorpion', 'black widow, Latrodectus mactans', 'tarantula', 'centipede',
                        'goose', 'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus', 'jellyfish',
                        'brain coral', 'snail', 'slug', 'sea slug, nudibranch', 'American lobster, Northern lobster, Maine lobster, Homarus americanus',
                        'spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish', 'black stork, Ciconia nigra',
                        'king penguin, Aptenodytes patagonica', 'albatross, mollymawk', 'dugong, Dugong dugon',
                        'Chihuahua', 'Yorkshire terrier', 'golden retriever', 'Labrador retriever',
                        'German shepherd, German shepherd dog, German police dog, alsatian', 'standard poodle',
                        'tabby, tabby cat', 'Persian cat', 'Egyptian cat', 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor',
                        'lion, king of beasts, Panthera leo', 'brown bear, bruin, Ursus arctos',
                        'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle', 'fly', 'bee',
                        'grasshopper, hopper', 'walking stick, walkingstick, stick insect', 'cockroach, roach',
                        'mantis, mantid', "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk",
                        'monarch, monarch butterfly, milkweed butterfly, Danaus plexippus', 'sulphur butterfly, sulfur butterfly',
                        'sea cucumber, holothurian', 'guinea pig, Cavia cobaya', 'hog, pig, grunter, squealer, Sus scrofa',
                        'ox', 'bison', 'bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis',
                        'gazelle', 'Arabian camel, dromedary, Camelus dromedarius', 'orangutan, orang, orangutang, Pongo pygmaeus',
                        'chimpanzee, chimp, Pan troglodytes', 'baboon', 'African elephant, Loxodonta africana',
                        'lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens', 'abacus',
                        "academic gown, academic robe, judge's robe", 'altar', 'apron',
                        'backpack, back pack, knapsack, packsack, rucksack, haversack', 'bannister, banister, balustrade, balusters, handrail',
                        'barbershop', 'barn', 'barrel, cask', 'basketball', 'bathtub, bathing tub, bath, tub',
                        'beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon',
                        'beacon, lighthouse, beacon light, pharos', 'beaker', 'beer bottle', 'bikini, two-piece',
                        'binoculars, field glasses, opera glasses', 'birdhouse', 'bow tie, bow-tie, bowtie',
                        'brass, memorial tablet, plaque', 'broom', 'bucket, pail', 'bullet train, bullet',
                        'butcher shop, meat market', 'candle, taper, wax light', 'cannon', 'cardigan',
                        'cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM',
                        'CD player', 'chain', 'chest', 'Christmas stocking', 'cliff dwelling', 'computer keyboard, keypad',
                        'confectionery, confectionary, candy store', 'convertible', 'crane', 'dam, dike, dyke',
                        'desk', 'dining table, board', 'drumstick', 'dumbbell', 'flagpole, flagstaff', 'fountain',
                        'freight car', 'frying pan, frypan, skillet', 'fur coat', 'gasmask, respirator, gas helmet',
                        'go-kart', 'gondola', 'hourglass', 'iPod', 'jinrikisha, ricksha, rickshaw', 'kimono',
                        'lampshade, lamp shade', 'lawn mower, mower', 'lifeboat', 'limousine, limo', 'magnetic compass',
                        'maypole', 'military uniform', 'miniskirt, mini', 'moving van', 'nail', 'neck brace',
                        'obelisk', 'oboe, hautboy, hautbois', 'organ, pipe organ', 'parking meter', 'pay-phone, pay-station',
                        'picket fence, paling', 'pill bottle', "plunger, plumber's helper", 'pole',
                        'police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria', 'poncho',
                        'pop bottle, soda bottle', "potter's wheel", 'projectile, missile', 'punching bag, punch bag, punching ball, punchball',
                        'reel', 'refrigerator, icebox', 'remote control, remote', 'rocking chair, rocker',
                        'rugby ball', 'sandal', 'school bus', 'scoreboard', 'sewing machine', 'snorkel', 'sock',
                        'sombrero', 'space heater', "spider web, spider's web", 'sports car, sport car',
                        'steel arch bridge', 'stopwatch, stop watch', 'sunglasses, dark glasses, shades',
                        'suspension bridge', 'swimming trunks, bathing trunks', 'syringe', 'teapot', 'teddy, teddy bear',
                        'thatch, thatched roof', 'torch', 'tractor', 'triumphal arch', 'trolleybus, trolley coach, trackless trolley',
                        'turnstile', 'umbrella', 'vestment', 'viaduct', 'volleyball', 'water jug', 'water tower',
                        'wok', 'wooden spoon', 'comic book', 'plate', 'guacamole', 'ice cream, icecream',
                        'ice lolly, lolly, lollipop, popsicle', 'pretzel', 'mashed potato', 'cauliflower',
                        'bell pepper', 'mushroom', 'orange', 'lemon', 'banana', 'pomegranate', 'meat loaf, meatloaf',
                        'pizza, pizza pie', 'potpie', 'espresso', 'alp', 'cliff, drop, drop-off', 'coral reef',
                        'lakeside, lakeshore', 'seashore, coast, seacoast, sea-coast', 'acorn']
        
        
    def __len__(self):
        """length
        Returns:
            self.num: length of dataset
        """
        return self.num
        
        
    def __getitem__(self, idx):
        #trans = transforms.Compose([transforms.ToTensor(), transforms.ToPILImage()])
        #out_data = trans(self.data[idx])
        out_data = self.data[idx]
        out_label = self.targets[idx]

        out_data = self.transform(out_data)
        
        if self.target_transform is not None:
            out_label = self.target_transform(out_label)

        return out_data, out_label

    
    
#The dataset for olivetti faces
class OlivettiFaces(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        """OlivettiFaces dataset for pytorch.
        You have to prepare dataset yourself by sklearn before use.
        
        Args:
            root: directory path for dataset images. (included train and test)
            transform: torchvision transforms for image. Defaults to None.
        """
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform
            
        self.data = fetch_olivetti_faces(data_home = root)
        self.num = len(self.data.data)
        
        
    def __len__(self):
        return self.num
        
        
    def __getitem__(self, idx):
        out_data = self.transform(self.data.images[idx])
        out_label = self.data.target[idx]

        return out_data, out_label

    
    
# The dataset for Glove
class Glove(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        """Glove dataset for pytorch.
        You have to preapre the dataset yourself before use.
        
        Args:
            root: directory path for dataset images. (included train and test)
            transform: torchvision transforms for image. Defaults to None.
        """
        root = os.path.expanduser(root)
        
        if transform is None:
            self.transform = torch.tensor
        else:
            self.transform = transform
            
        self.data = np.load(root + '/data.npy')[:10000]
        f = open(root + '/label.txt')
        labels = f.readline()
        labels = labels.rstrip('\n').split(' ')
        self.labels = labels[:10000]
        self.num = len(self.data)
    
    
    def __len__(self):
        return self.num
    
    
    def __getitem__(self, idx):
        out_data = self.transform(self.data[idx])
        out_label = self.labels[idx]
        
        return out_data, out_label
    
    
    
# dataset with returing index
class IndexDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        """This IndexDataset is wrapper of pytorch dataset.
        This returns common items and index as one iterative.
        
        Args:
            dataset: pytorch dataset 
        """
        self.dataset = dataset
        self.classes = dataset.classes
    
    
    def __len__(self):
        return len(self.dataset)
    
    
    def __getitem__(self, idx):
        return idx, self.dataset[idx][0], self.dataset[idx][1]
        

        
# Subset Dataset
class SubDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, sample_index=None):
        """This SubDataset is wrapper of pytorch dataset.
        This uses subset of dataset with sample_index.
        
        Args:
            dataset: pytorch dataset 
            sample_index: Sample index for subset. Defaults to None.
        """
        self.dataset = dataset
        if sample_index is None:
            sample_index = range(len(dataset))
        self.sample_index = sample_index
        self.len_original_dataset = len(dataset)
        self.tmp = sample_index
        self.classes = dataset.classes
        
        
    def __len__(self):
        return len(self.sample_index)
    
    
    def __getitem__(self, idx):
        sub_idx = self.sample_index[idx]
        return self.dataset[sub_idx]
    
    
    def reset_sample_index(self, sample_index):
        """Reset the subset.
        
        Args:
            sample_index: Sample index for subset.
        """
        self.sample_index = sample_index
    
    
    def train(self):
        """For train when you want to use subset.
        """
        self.sample_index = self.tmp
        
    
    def test(self):
        """For test when you wanto to use whole dataset.
        """
        self.tmp = self.sample_index
        self.sample_index = range(self.len_original_dataset)
        

        
class RouletteSampler(BatchSampler):
    def __init__(
        self,
        sampler,
        batch_size,
        drop_last=False,
        sample_weight=None,
        sample_num=None,
    ):
        """This is batch sampler for pytorch dataloader.
        When sampling, dataloader samples mini-batch samples with sample_weight.
        The higher sample_weight, the more likely it is to be selected.
        
        Args:
            sampler: Dataset indexes. e.g. range(len(dataset)).
            batch_size: Batach size.
            drop_last: Drop last mini-batch. Defaults to Faluse.
            sample_weight: Sample weights. Defaults to None.
            sample_num: The number of samples.
        """
        super(RouletteSampler, self).__init__(sampler, batch_size, drop_last)
        self.sample_weight = sample_weight
        if sample_num is None:
            self.sample_num = len(self.sampler)
        else:
            self.sample_num = sample_num

            
    def __iter__(self):
        if self.sample_weight is None:
            batch = []
            sampler = list(RandomSampler(self.sampler))
            for idx in sampler:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if len(batch) > 0 and not self.drop_last:
                yield batch
        else:
            roulette_sampling = torch.multinomial(self.sample_weight, self.sample_num, replacement=False)
            batch = []
            for idx in roulette_sampling:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if len(batch) > 0 and not self.drop_last:
                yield batch
            
    
    def reset_weight(self, sample_weight):
        """Reset the sample weights
        Args:
            sample_weight: Sample weights.
        """
        self.sample_weight = sample_weight
        

        
class ReplaceRouletteSampler(BatchSampler):
    def __init__(
        self,
        sampler,
        batch_size,
        drop_last=False,
        sample_weight=None,
        sample_num=None,
    ):
        """This is batch sampler for pytorch dataloader.
        When sampling, dataloader samples mini-batch samples with sample_weight.
        The higher sample_weight, the more likely it is to be selected.
        This sampling allows replacement.
        
        Args:
            sampler: Dataset indexes. e.g. range(len(dataset)).
            batch_size: Batach size.
            drop_last: Drop last mini-batch. Defaults to Faluse.
            sample_weight: Sample weights. Defaults to None.
            sample_num: The number of samples.
        """
        super(ReplaceRouletteSampler, self).__init__(sampler, batch_size, drop_last)
        self.sample_weight = sample_weight
        if sample_num is None:
            self.sample_num = len(self.sampler)
        else:
            self.sample_num = sample_num

            
    def __iter__(self):
        if self.sample_weight is None:
            batch = []
            sampler = list(RandomSampler(self.sampler))
            for idx in sampler:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if len(batch) > 0 and not self.drop_last:
                yield batch
        else:
            count = 0
            while count < len(self.sampler):
                batch = torch.multinomial(self.sample_weight, self.batch_size, replacement=False)
                yield batch
                count += len(batch)
            
    
    def reset_weight(self, sample_weight):
        """Reset the sample weights
        Args:
            sample_weight: Sample weights.
        """
        self.sample_weight = sample_weight
    

    
class SiameseDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        """This SiameseDataset is wrapper of pytorch dataset.
        This returns 2 samples of dataset for Siamese network.
        
        Args:
            dataset: pytorch dataset.
        """
        self.dataset = dataset
        self.targets = dataset.targets
        
        if not isinstance(self.targets, (np.ndarray, np.generic)):
                self.targets = np.array(self.targets)
        
        self.labels_set = set(self.targets) #make label set 0-9
        self.label_to_indices = {label : np.where(self.targets == label)[0] for label in self.labels_set}
        
        if not self.dataset.train:
            random_state = np.random.RandomState(29)
            
            positive_pairs = []
            positive_target = 1
            for i in range(0, len(self.targets), 2):
                positive_label_indices = self.label_to_indices[self.targets[i].item()]
                random_positive_pair = random_state.choice(positive_label_indices)
                positive_pairs.append([i, random_positive_pair, positive_target])
            
            negative_pairs = []
            negative_target = 0
            for i in range(0, len(self.targets), 2):
                negative_label = np.random.choice(list(self.labels_set - set([self.targets[i].item()])))
                negative_label_indices = self.label_to_indices[negative_label]
                random_negative_pair = random_state.choice(negative_label_indices)
                negative_pairs.append([i, random_negative_pair, negative_target])
            
            self.test_pairs = positive_pairs + negative_pairs
    
    
    def __getitem__(self, index):
        if self.dataset.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.dataset[index]
            
            if target == 1:
                pair_index = index
                while pair_index == index:
                    positive_label_indices = self.label_to_indices[label1]
                    pair_index = np.random.choice(positive_label_indices)
            else:
                negative_label = np.random.choice(list(self.labels_set - set([label1])))
                negative_label_indices = self.label_to_indices[negative_label]
                pair_index = np.random.choice(negative_label_indices)
            
            img2, label2 = self.dataset[pair_index]
        else:
            img1, label1 = self.dataset[self.test_pairs[index][0]]
            img2, label2 = self.dataset[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]
            
        return (img1, img2), (label1, label2), target
    
    
    def __len__(self):
        return len(self.dataset)

    

class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        """This TripletDataset is wrapper of pytorch dataset.
        This returns 3 samples of dataset for Triplet network.
        
        Args:
            dataset: pytorch dataset.
        """
        self.dataset = dataset
        
        if self.dataset.train:
            self.train_labels = self.dataset.targets
            if not isinstance(self.train_labels, (np.ndarray, np.generic)):
                self.train_labels = np.array(self.train_labels)
            
            self.train_data = self.dataset.data
            self.labels_set = set(self.train_labels) #make label set 0-9
            self.label_to_indices = {label : np.where(self.train_labels == label)[0] for label in self.labels_set}

        else:
            self.test_labels = self.dataset.targets
            if not isinstance(self.test_labels, (np.ndarray, np.generic)):
                self.test_labels = np.array(self.test_labels)
            
            self.test_data = self.dataset.data
            
            self.labels_set = set(self.test_labels) #make label set 0-9
            self.label_to_indices = {label : np.where(self.test_labels == label)[0] for label in self.labels_set}
            
            random_state = np.random.RandomState(29)
        
            triplets = [
                [
                    i,
                    random_state.choice(self.label_to_indices[self.test_labels[i]]),
                    random_state.choice(
                        self.label_to_indices[
                            np.random.choice(
                                list(
                                    self.labels_set - set([self.test_labels[i]])
                                )
                            )
                        ]
                    )
                ]
                for i in range(len(self.test_data))
            ]
            
            self.test_triplets = triplets
            
            
    def __getitem__(self, index):
        if self.dataset.train:
            img1, label1 = self.dataset[index]
            if not isinstance(label1, torch.Tensor):
                label1 = torch.tensor(label1)
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1.item()])
            
            negative_label = np.random.choice(list(self.labels_set - set([label1.item()])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2, label2 = self.dataset[positive_index]
            img3, label3 = self.dataset[negative_index]
            
        else:
            img1, label1 = self.dataset[self.test_triplets[index][0]]
            img2, label2 = self.dataset[self.test_triplets[index][1]]
            img3, label3 = self.dataset[self.test_triplets[index][2]]
        
        return (img1, img2, img3), (label1, label2, label3)
    
    
    def __len__(self):
        return len(self.dataset)
    
    

class KDTripletDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        """This KDTripletDataset is wrapper of pytorch dataset.
        This returns 2 samples of dataset for Knowledge Triplet network.
        
        Args:
            dataset: pytorch dataset.
        """
        self.dataset = dataset
        
        data = self.dataset.data
        labels = self.dataset.targets
        if not isinstance(labels, (np.ndarray, np.generic)):
            labels = np.array(labels)
        
        # make label set 0-9
        labels_set = set(labels)
        
        # make the indices excepted each classes
        label_to_indices = {label : np.where(labels.numpy() != label)[0] for label in labels_set}
        
        if self.dataset.train:
            self.negative_indices = label_to_indices
        else:
            self.negative_indices = [[np.random.choice(label_to_indices[labels[i].item()])] for i in range(len(data))]
        

    def __getitem__(self, index):
        if self.dataset.train:
            img1_2, label1_2 = self.dataset[index]
            if not isinstance(label1_2, torch.Tensor):
                label1_2 = torch.tensor(label1_2)
            img3, label3 = self.dataset[np.random.choice(self.negative_indices[label1_2.item()])]
        else:
            img1_2, label1_2 = self.dataset[index]
            img3, label3 = self.dataset[self.negative_indices[index][0]]
        
            
        return (img1_2, img3), (label1_2, label3)
    
    
    def __len__(self):
        return len(self.dataset)
    
    
    
class Datasets(object):
    def __init__(
        self,
        dataset_name,
        batch_size=100,
        num_workers=2,
        transform=None,
        test_transform="same",
        shuffle=True,
        batch_sampler=None,
        dataset_dir="~/",
    ):
        """Datasets utility.
        You have to prepare some datasets yourself before use,
        because official torch doesnot support some datasets (e.g. TinyImagenet).
        
        This returns torch dataloader, torch dataset, dataset classes, input channels and so on.
        Especcially, it can take dataset from string dataset name.
        
        Args:
            dataset_name(str): dataset name. MNIST, FashionMNIST, CIFAR10, CIFAR100, STL10, TinyImagenet
                                OlivettiFaces, COIL-20, Glove, VGGface, FractalDB-60, and FractalDB-1k.
            batch_size: Batch size for torch dataloader. Defaults to 100.
            num_workers: The number of workers for torch dataloader. Defaults to 2.
            transform: Data transoform for train data. Defaults to None.
            test_transform: Data transform for test data. Defaults to "same".
            shuffle: Shuffle for sampling. Defaults to True.
            batch_sampler: Batch sampler for torch dataloader. Defaults to None.
            dataset_dir: Directory path for datasets. Defaults to "~/".
        """
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.shuffle = shuffle
        self.batch_sampler = batch_sampler
        self.dataset_dir = dataset_dir
        if test_transform == "same":
            self.test_transform = transform
        else:
            self.test_transform = test_transform
            
        if self.transform is None:
                self.transform = transforms.Compose([transforms.ToTensor()])
        if self.test_transform is None:
                self.test_transform = transforms.Compose([transforms.ToTensor()])
        
        
    def create(self, options = []):
        print("Dataset :",self.dataset_name)
        
        trainset, classes, base_labels, input_channels = self.set_dataset(train=True)
        testset, _, _, _ = self.set_dataset(train=False)
        
        for opt in options:
            if opt == 'Index':
                trainset = IndexDataset(trainset)
                testset = IndexDataset(testset)
            elif opt == 'Subset':
                trainset = SubDataset(trainset)
                testset = SubDataset(testset)
            elif opt == 'Triplet':
                trainset = TripletDataset(trainset)
                testset = TripletDataset(testset)
            elif opt == 'Siamese':
                trainset = SiameseDataset(trainset)
                testset = SiameseDataset(testset)
            elif opt == 'KDTriplet':
                trainset = KDTripletDataset(trainset)
                testset = KDTripletDataset(testset)
            elif opt == 'Roulette':
                self.batch_sampler = RouletteSampler(range(len(trainset)), self.batch_size)
            elif opt == 'ReplaceRoulette':
                self.batch_sampler = ReplaceRouletteSampler(range(len(trainset)), self.batch_size)
            else:
                print(f"No options {opt}.")
        
        trainloader = self.set_dataloader(trainset)
        
        if testset is not None:
            self.batch_sampler = None
            testloader = self.set_dataloader(
                testset,
                shuffle=False,
                batch_sampler=None,
            )
        else:
            testloader = None
            
        return [trainloader, testloader, classes, base_labels, input_channels, trainset, testset]
    
    
    def set_dataloader(
        self,
        dataset,
        num_workers=None,
        batch_size=None,
        shuffle=None,
        batch_sampler=None,
    ):
        """Datasets utility.        
        This returns torch dataloader.
        
        Args:
            dataset: torch dataset.
            num_workers: The number of workers for torch dataloader. Defaults to None.
            batch_size: Batch size for torch dataloader. Defaults to None.
            shuffle: Shuffle for sampling. Defaults to None.
            batch_sampler: Batch sampler for torch dataloader. Defaults to None.
        """
        if num_workers is None:
            num_workers = self.num_workers
        if batch_size is None:
            batch_size = self.batch_size
        if shuffle is None:
            shuffle = self.shuffle
        if batch_sampler is None:
            batch_sampler = self.batch_sampler
        
        if batch_sampler is not None:
            dataloader  =torch.utils.data.DataLoader(
                dataset,
                batch_sampler = batch_sampler,
                num_workers=num_workers,
            )
        else:
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
            )
        return dataloader
    
    def set_dataset(self, train=True, trans=None):
        """Datasets utility.        
        This returns torch dataset.
        If test dataset is None, it returns None for testset 
        
        Args:
            train: Get train dataset. Defaults to True.
            trans: Data transform. Defaults to None.
        """
        if train:
            print("set train data")
            transform = self.transform
        else:
            print("set test data")
            transform = self.test_transform
        
        if trans is not None:
            transform = trans
        
        if self.dataset_name == "MNIST":
            path = os.path.join(self.dataset_dir, 'MNISTDataset/data')
            dataset = torchvision.datasets.MNIST(
                root=path,
                train=train,
                download=True,
                transform=transform,
            )
            classes = list(range(10))
            base_labels = dataset.classes
            input_channels = 1
            
        elif self.dataset_name == "FashionMNIST":
            path = os.path.join(self.dataset_dir, 'FashionMNISTDataset/data')
            dataset = torchvision.datasets.FashionMNIST(
                root=path,
                train=train,
                download=True,
                transform=transform,
            )
            classes = list(range(10))
            base_labels = dataset.classes
            input_channels = 1
            
        elif self.dataset_name == "CIFAR10":
            path = os.path.join(self.dataset_dir, 'CIFAR10Dataset/data')
            dataset = torchvision.datasets.CIFAR10(
                root=path,
                train=train,
                download=True,
                transform=transform,
            )
            classes = list(range(10))
            base_labels = dataset.classes
            input_channels = 3
            
        elif self.dataset_name == "CIFAR100":
            path = os.path.join(self.dataset_dir, 'CIFAR100Dataset/data')
            dataset = torchvision.datasets.CIFAR100(
                root=path,
                train=train,
                download=True,
                transform=transform,
            )
            classes = list(range(100))
            base_labels = dataset.classes
            input_channels = 3
            
        elif self.dataset_name == "STL10":
            path = os.path.join(self.dataset_dir, 'STL10/data')
            if train:
                split = "train"
            else:
                split = "test"
            dataset = torchvision.datasets.STL10(
                root=path,
                split=split,
                download=True,
                transform=transform,
            )
            classes = list(range(10))
            base_labels = dataset.classes
            input_channels = 3
        
        elif self.dataset_name == "TinyImagenet":
            path = os.path.join(self.dataset_dir, 'TinyImagenet')
            dataset = TinyImagenet(
                root=path,
                train=train,
                transform=transform,
            )
            classes = list(range(200))
            base_labels = dataset.classes
            input_channels = 3
            
        elif self.dataset_name == "OlivettiFaces":
            path = os.path.join(self.dataset_dir, 'Olivettifaces/data')
            dataset = OlivettiFaces(
                root=path,
                transform=transform,
            )
            classes = list(range(40))
            base_labels = []
            input_channels = 1
            if not train:
                dataset = None
            
        elif self.dataset_name == "COIL-20":
            path = os.path.join(self.dataset_dir, 'COIL-20/data')
            transform = transforms.Compose(
                [
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                ],
            )
            dataset = torchvision.datasets.ImageFolder(
                root=path,
                transform=transform,
            )
            classes = list(range(20))
            base_labels = dataset.classes
            input_channels = 1
            if not train:
                dataset = None
        
        elif self.dataset_name == "Glove":
            path = os.path.join(self.dataset_dir, 'Glove/data')
            dataset = Glove(
                root=path,
                transform=None,
            )
            classes = list(range(len(dataset)))
            base_labels = dataset.labels
            input_channels = 1
            if not train:
                dataset = None
            
        elif self.dataset_name == "VGGface":
            if train:
                path = os.path.join(self.dataset_dir, 'VGGface2/train')
            else:
                path = os.path.join(self.dataset_dir, 'VGGface2/test')
            dataset = torchvision.datasets.ImageFolder(
                root=path,
                transform=transform,
            )
            classes = None
            base_labels = None
            input_channels = 3
        
        elif self.dataset_name == "FractalDB-60":
            path = os.path.join(self.dataset_dir, 'FractalDB-60/data')
            dataset = torchvision.datasets.ImageFolder(
                root=path,
                transform=transform,
            )
            classes = list(range(60))
            base_labels = dataset.classes
            input_channels = 1
            if not train:
                dataset = None
        
        elif self.dataset_name == "FractalDB-1k":
            path = os.path.join(self.dataset_dir, 'FractalDB-1k/data')
            dataset = torchvision.datasets.ImageFolder(
                root=path,
                transform=transform,
            )
            classes = list(range(1000))
            base_labels = dataset.classes
            input_channels = 1
            if not train:
                dataset = None
            
        else:
            raise KeyError("Unknown dataset: {}".format(self.dataset_name))
            
        return dataset, classes, base_labels, input_channels
        
    
    def worker_init_fn(self, worker_id):                                                          
        np.random.seed(worker_id)