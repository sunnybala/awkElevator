import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import matplotlib.patches as patches
import scipy.spatial.distance as dist
import matplotlib
from matplotlib import collections as mc

peeps =pd.DataFrame(np.random.random([3,2]),columns=['x','y'],index=['A','B','C'])
pwd = {}

def main(intent,n_people=5,METHOD='nearest'):
	fig = plt.figure(figsize=(7, 7))
	ax = fig.add_axes([0, 0, 1, 1], frameon=False)
	
	BOUNDS = 1
	SIZE = 10000
	SIZE = 2500
	# METHOD = "rand"
	# METHOD = 'midpoint'
	# METHOD = 'wdistderiv'
	# METHOD = 'nearest'
	# strings
	SPD = "Sum Pairwise Dist: "
	MPD = "Min Pairwise Dist: "

	# set up axes
	ax.set_xlim(-BOUNDS*1.25, BOUNDS*1.25), ax.set_xticks([])
	ax.set_ylim(-BOUNDS*1.5, BOUNDS*1.25), ax.set_yticks([])

	# Create people data
	people = np.zeros(n_people, dtype=[('position', float, 2),
	                                      ('size',     float, 1),
	                                      ('growth',   float, 1),
	                                      ('color',    float, 4)])

	# Initialize the people in random positions and with
	# random growth rates.
	people['position'] = np.random.uniform(-BOUNDS*.9, BOUNDS*.9, (n_people, 2))
	# Construct the scatter which we will update during animation
	# as the people develop.
	ax.add_patch(
	    patches.Rectangle(
	        (BOUNDS, -BOUNDS),
	        -BOUNDS*2,
	        BOUNDS*2,
	        fill=False 
	    )
	)

	# get unique colors
	colormap = plt.cm.gist_ncar
	colorst = [colormap(i) for i in np.linspace(.1,.9,n_people)]
	scat = ax.scatter(people['position'][:, 0], people['position'][:, 1],
	                  s=SIZE, lw=0.5, c=colorst)# edgecolors=people['color']

	obj = sum(dist.pdist(people['position'])).round(4)
	objText = ax.text(-.2, -BOUNDS*1.2, SPD+str(obj), fontsize=12,bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
	objText2 = ax.text(-.2, -BOUNDS*1.4, MPD+str(obj), fontsize=12,bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})

	def getStep(idx,method='rand'):
		if method == 'rand':
			step = np.random.uniform(-1, 1, 2) 
		elif method == 'midpoint':
			midpt = people['position'].mean(axis=0)
			step = people['position'][idx] - midpt
			#step = .05*step
		elif method == 'nearest':
			nDist = 1000
			nIdx = idx
			for ii in range(n_people):
				if ii != idx and dist.euclidean(people['position'][idx],people['position'][ii]) < nDist:
					nDist = dist.euclidean(people['position'][idx],people['position'][ii])
					nIdx = ii
					midpt = people['position'][nIdx]
			step = people['position'][idx] - midpt
		# elif method == 'distderiv':
		# 	# Derivative of the distance formula
		# 	a = people['position'][idx]
		# 	sx = 0
		# 	sy = 0
		# 	for b in people['position']:
		# 		if all(b != a):
		# 			sx += (a[0]-b[0])/np.sqrt((a[0]-b[0])**2+(a[1] - b[1])**2)
		# 			sy += (a[1]-b[1])/np.sqrt((a[0]-b[0])**2+(a[1] - b[1])**2)
		# 	step = np.array([sx,sy])
		# elif method == 'wdistderiv':
		# 	# Similar to distance formula above except weighted by 1/distance
		# 	# This is because you probably care a lot more about moving an inch away 
		# 	# from people really close to you vs. moving an inch away from someone 
		# 	# further away 
		# 	a = people['position'][idx]
		# 	sx = 0
		# 	sy = 0
		# 	for b in people['position']:
		# 		if all(b != a):
		# 			w = dist.euclidean(a,b)
		# 			w = max(1./w,100)
		# 			sx += w*(a[0]-b[0])/np.sqrt((a[0]-b[0])**2+(a[1] - b[1])**2)
		# 			sy += w*(a[1]-b[1])/np.sqrt((a[0]-b[0])**2+(a[1] - b[1])**2)
		# 	step = np.array([sx,sy])
		return step

	def update(frame_number):
	    # Get an index which we can use to re-spawn the oldest people.
	    current_index = frame_number % n_people

	    steps = {}
	    lines = []
	    oldPos = people['position'].copy()

	    noise = .01 if frame_number < 1000 else float(10)/frame_number**2
	    print(noise)
	    for idx in range(n_people):
		    steps[idx] = getStep(idx,METHOD)
		    if intent:
		    	lines.append([people['position'][idx],people['position'][idx]+steps[idx]])
		    steps[idx] *= .02 
		    steps[idx] += (2.0*np.random.random(2) - 1.0)*noise
		    
		    # print(idx,steps[idx])

	    for idx in steps:
		    people['position'][idx] += steps[idx]
		    people['position'][idx] = np.clip(people['position'][idx],-BOUNDS+np.sqrt(SIZE)/315.,BOUNDS-np.sqrt(SIZE)/315.)
	    
	    # print(people['position'])
	    if (oldPos == people['position']).all():
	    	assert False

	    # Update the scatter collection, with the new colors, sizes and positions.
	    scat.set_edgecolors(people['color'])
	    #scat.set_sizes(people['size'])
	    scat.set_offsets(people['position'])

	    # update and print objective
	    pwd[frame_number] = sum(dist.pdist(people['position']))
	    obj = pwd[frame_number].round(3)
	    mpdn = min(dist.pdist(people['position'])).round(3)
	    objText.set_text(SPD + str(obj))
	    objText2.set_text(MPD + str(mpdn))



	# Construct the animation, using the update function as the animation
	# director.
	animation = FuncAnimation(fig, update, interval=10)
	plt.show()