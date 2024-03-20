import numpy as np

def load_bop_results(path, version='bop19'):
  """Loads 6D object pose estimates from a file.

  :param path: Path to a file with pose estimates.
  :param version: Version of the results.
  :return: List of loaded poses.
  """
  results = []

  # See docs/bop_challenge_2019.md for details.
  if version == 'bop19':
    header = 'scene_id,im_id,obj_id,score,R,t,time'
    with open(path, 'r') as f:
      line_id = 0
      for line in f:
        line_id += 1
        if line_id == 1 and header in line:
          continue
        else:
          elems = line.split(',')
          if len(elems) != 7:
            raise ValueError(
              'A line does not have 7 comma-sep. elements: {}'.format(line))

          result = {
            'scene_id': int(elems[0]),
            'im_id': int(elems[1]),
            'obj_id': int(elems[2]),
            'score': float(elems[3]),
            'R': np.array(
              list(map(float, elems[4].split())), np.float).reshape((3, 3)),
            't': np.array(
              list(map(float, elems[5].split())), np.float).reshape((3, 1)),
            'time': float(elems[6])
          }

          results.append(result)
  else:
    raise ValueError('Unknown version of BOP results.')

  return results