from segmentation_functions import *

stories = ['Run', 'GoHitler', 'MyMothersLife']
tempuratures = [0, 0.5, 1]
iterations = 20

for story in stories:
     for temp in tempuratures:
          gpt_events = gpt_segmentation(f'{story}.txt', iters=iterations, model='gpt-4', temperature=temp)
          llama_events = llama_segmentation(f'{story}.txt', iters=iterations, temperature=temp)

          for i in range(iterations):
               np.save(f'GPT/{story}_events_{temp}_{i}.npy', gpt_events[i], allow_pickle=True)
               np.save(f'LLaMA/{story}_events_{temp}_{i}.npy', llama_events[i], allow_pickle=True)