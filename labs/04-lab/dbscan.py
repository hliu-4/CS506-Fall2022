import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
from PIL import Image

centers = [[1, 1], [-1, -1], [1, -1]]
X, _ = datasets.make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)
plt.scatter(X[:,0],X[:,1],s=10, alpha=0.8)
plt.savefig('plt0.png')
img = Image.open('plt0.png') 
plt.close()


class DBC():

    def __init__(self, dataset, min_pts, epsilon, colors, img):
        self.dataset = dataset
        self.min_pts = min_pts
        self.epsilon = epsilon
        self.colors = colors
        self.imgs = []
        self.imgs.append(img)
        self.x = 1
        
    def _convert_gif(self, imgs, filename, duration):
        stacked_images = []
        for img in imgs:
            stacked_images.append(Image.fromarray(np.asarray(img)))
        
        stacked_images[0].save(
            filename + '.gif',
            optimize=False,
            save_all=True,
            append_images=stacked_images[1:],
            loop=0,
            duration=duration
        )
        
        return

    def eps_neighborhood(self, pt):
        neighborhood = []
        
        for i in range(len(self.dataset)):
            if np.linalg.norm(self.dataset[i] - self.dataset[pt]) <= self.epsilon:
                neighborhood.append(i)
                
        return neighborhood
    
    def make_image(self, assignments):
        plt.scatter(self.dataset[:, 0], self.dataset[:, 1], color=colors[assignments].tolist(), s=10, alpha=0.8)
        plt.savefig('plt'+str(self.x)+'.png')
        
        img = Image.open('plt'+str(self.x)+'.png')
        self.imgs.append(img)
        
        self.x += 1
        plt.close()
    
    def assign_core(self, pt, assignments, cluster):
        assignments[pt] = cluster
        
        self.make_image(assignments)
        
        neighborhood = self.eps_neighborhood(pt)
        
        while neighborhood:
            next_pt = neighborhood.pop()
            
            if assignments[next_pt] == cluster:
                continue
            
            assignments[next_pt] = cluster
            self.make_image(assignments)
            
            next_neighborhood = self.eps_neighborhood(next_pt)
            
            if len(next_neighborhood) >= self.min_pts:
                neighborhood += next_neighborhood
                assignments = self.assign_core(next_pt, assignments, cluster)
    
        return assignments

    def dbscan(self):
        """
        returns a list of assignments. The index of the
        assignment should match the index of the data point
        in the dataset.
        """
        # 0 means point hasn't been assigned yet
        assignments = [0 for _ in range(len(self.dataset))]
        cluster = 1
        
        for pt in range(len(self.dataset)):
            if assignments[pt] != 0:
                continue
            
            # define core point
            if len(self.eps_neighborhood(pt)) >= self.min_pts:
                # assign all core points of the same cluster
                assignments = self.assign_core(pt, assignments, cluster)
                cluster += 1
                
        self._convert_gif(self.imgs, 'animation', 50)
        
        return assignments

colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)
clustering = DBC(X, 3, .2, colors, img).dbscan()