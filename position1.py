# Miscellaneous routines (some interesting) related to covariance
# estimation. 

import util, signal, position

def compute_contour(x_hat, f, Q):
  ''' Find the points that fall within confidence region of the estimate. 
  
    Given a point x_hat known to be contained by a contour defined by
    f(x) < Q, compute the contour.  
  ''' 
  S = set(); S.add((x_hat[0], x_hat[1]))
  level_set = S.copy()
  contour = set()
  max_size = 10000 # FIXME Computational stop gap. 

  while len(S) > 0 and len(S) < max_size and len(level_set) < max_size: 
    R = set()
    for x in S:
      if f(x) < Q: 
        level_set.add(x)
        R.add((x[0]+1, x[1]-1)); R.add((x[0]+1, x[1])); R.add((x[0]+1, x[1]+1))
        R.add((x[0],   x[1]-1));                        R.add((x[0] ,  x[1]+1))
        R.add((x[0]-1, x[1]-1)); R.add((x[0]-1, x[1])); R.add((x[0]-1, x[1]+1)) 
      else: 
        contour.add(x)
    S = R.difference(level_set)

  if len(S) >= max_size or len(level_set) >= max_size: 
    return (None, None) # Unbounded confidence region
  return (level_set, contour)

def fit_ellipse(x, y): 
  ''' Fit ellipse parameters to a set of points in R^2. 
  
    The points should correspond a perfect ellipse. 
  ''' 
  x_lim = np.array([np.min(x), np.max(x)])
  y_lim = np.array([np.min(y), np.max(y)])
  
  x_center = np.array([np.mean(x_lim), np.mean(y_lim)])
 
  X = np.vstack((x,y))
  D = (lambda d: np.sqrt(
          (d[0] - x_center[0])**2 + (d[1] - x_center[1])**2))(X)
  x_major = x_center - X[:,np.argmax(D)] 
  angle = np.arctan2(x_major[1], x_major[0])
  axes = np.array([np.max(D), np.min(D)])
  return (x_center, angle, axes)

def fit_noisy_ellipse(x, y):
  ''' Least squares fit of an ellipse to a set of points in R^2. 

    The points are allowed to be noisy. Method due to  
    http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
  '''
  x = x[:,np.newaxis]
  y = y[:,np.newaxis]
  D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
  S = np.dot(D.T,D)
  C = np.zeros([6,6])
  C[0,2] = C[2,0] = 2; C[1,1] = -1
  E, V =  np.linalg.eig(np.dot(np.linalg.inv(S), C))
  n = np.argmax(np.abs(E))
  A = V[:,n]
     
  # Center of ellipse
  b,c,d,f,g,a = A[1]/2, A[2], A[3]/2, A[4]/2, A[5], A[0]
  num = b*b-a*c
  x0=(c*d-b*f)/num
  y0=(a*f-b*d)/num
  x = np.array([x0,y0])

  # Angle of rotation
  angle = 0.5*np.arctan(2*b/(a-c))

  # Length of Axes
  up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
  down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
  down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
  res1=np.sqrt(up/down1)
  res2=np.sqrt(up/down2)
  axes = np.array([res1, res2])

  return (x, angle, axes)


def fit_contour(x, y, N):
  ''' Fit closed countour to a set of points in R^2. 
  
    Convert the Cartesian coordinates (x, y) to polar coordinates (theta, r)
    and fit a spline. Sample uniform angles from this spline and compute the
    Fourier transform of their distancxe to the centroid of the contour. 
    `N` is the number of samples. 
  
    http://stackoverflow.com/questions/13604611/how-to-fit-a-closed-contour
  '''
  x0, y0 = np.mean(x), np.mean(y)
  C = (x - x0) + 1j * (y - y0)
  angles = np.angle(C)
  distances = np.abs(C)
  sort_index = np.argsort(angles)
  angles = angles[sort_index]
  distances = distances[sort_index]
  angles = np.hstack(([ angles[-1] - 2*np.pi ], angles, [ angles[0] + 2*np.pi ]))
  distances = np.hstack(([distances[-1]], distances, [distances[0]]))

  f = spline1d(angles, distances)
  theta = scipy.linspace(-np.pi, np.pi, num=N, endpoint=False) 
  distances_uniform = f(theta)

  fft_coeffs = np.fft.rfft(distances_uniform)
  fft_coeffs[5:] = 0 
  r = np.fft.irfft(fft_coeffs)
 
  x_fit = x0 + r * np.cos(theta)
  y_fit = y0 + r * np.sin(theta)

  return (x_fit, y_fit)






  


  

 



### Testing, testing ... ######################################################


def test1(): 

  import time
  
  cal_id = 3
  dep_id = 105
  t_start = 1407452400 
  t_end = 1407455985 - (59 * 60) 

  db_con = util.get_db('reader')
  sv = signal1.SteeringVectors(db_con, cal_id)
  signal = signal1.Signal(db_con, dep_id, t_start, t_end)

  sites = util.get_sites(db_con)
  (center, zone) = util.get_center(db_con)
  assert zone == util.get_utm_zone(db_con)
  
  start = time.time()
  pos = PositionEstimator(dep_id, sites, center, signal, sv, 
    method=signal1.Signal.MLE)
  print "Finished in {0:.2f} seconds.".format(time.time() - start)
 
  print compute_conf(pos.p, pos.num_sites, sites, pos.splines)
  


if __name__ == '__main__':
  

  #test_exp()
  #test_bearing()
  #test_mle()
  test1()
