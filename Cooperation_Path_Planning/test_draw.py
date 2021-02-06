import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

plt.figure(figsize=(10,10))
plt.xlim(1,13)
plt.ylim(0,105)
plt.title("size=24x24")
plt.xlabel("pool step")
plt.ylabel("redundant path-value cost and time saved: %")
time_and_cost = [[2,93.67,35.86],[3,98.01,35.03],[4,98.37,30.69],[6,97.16,23.13],[8,94.64,17.99],[12,83.27,9.06]]
first = True
for p in time_and_cost:
    plt.plot(p[0],p[1],'o',color='r')
    plt.text(p[0]-0.5,p[1]+1.5,'%.2f'%p[1]+'%',color='r')

    plt.plot(p[0],p[2],'o',color='g')
    plt.text(p[0]-0.3,p[2]+1.5,'%.2f'%p[2]+'%',color='g')

for i in range(len(time_and_cost)-1):
    start = (time_and_cost[i][0],time_and_cost[i+1][0])
    end = (time_and_cost[i][1],time_and_cost[i+1][1])
    plt.plot(start,end,color='r')

    start = (time_and_cost[i][0],time_and_cost[i+1][0])
    end = (time_and_cost[i][2],time_and_cost[i+1][2])
    plt.plot(start,end,color='g')

plt.show()