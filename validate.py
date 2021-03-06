# beacon.dat       -- t_win=15
# beacon136.dat    -- include=[1,3,6]
# beacon13.dat     -- include=[1,3]
# beacon16.dat     -- include=[1,6]
# beacon36.dat     -- include=[3,6]
# f0beacon.dat     -- score_threshold>=0, t_win=15
# f0.5beacon.dat   -- score_threshold>=0.5, t_win=60
# f30beacon.dat    -- score_threshold>=0, t_win=30
# f60beacon.dat    -- score_threshold>=0, t_win=60
# f60beacon.e1.dat -- score_threshold>=0, t_win=60, exclude=[1]

from qraat.srv import util, signal, position

import scipy.stats
import numpy as np
import matplotlib.pyplot as pp
import pickle

suffix = 'dat'
conf_level=0.95

position.NORMALIZE_SPECTRUM=False


def band_dist(dep_id, sites, db_con):
  
  cur = db_con.cursor()
  band3 = {}
  band10 = {}
  for site_id in sites.keys():
    cur.execute('''SELECT band3, band10 
                     FROM qraat.est
                    WHERE deploymentID=%s
                      AND siteID=%s''', (dep_id, site_id))
    band3[site_id] = []
    band10[site_id] = []
    for (bw3, bw10) in cur.fetchall(): 
      band3[site_id].append(bw3)
      band10[site_id].append(bw10)
    
    N=100

    fig = pp.gcf()
    n, bins, patches = pp.hist(band3[site_id], N, histtype='stepfilled')
    pp.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
    pp.savefig('band3_dep%d_site%d.png' % (dep_id, site_id))
    pp.clf()
    
    fig = pp.gcf()
    n, bins, patches = pp.hist(band10[site_id], N, histtype='stepfilled')
    pp.setp(patches, 'facecolor', 'r', 'alpha', 0.75)
    pp.savefig('band10_dep%d_site%d.png' % (dep_id, site_id))
    pp.clf()


def process(sv, sites, center, params):

  dep_id = params['dep_id']
  t_start = params['t_start']
  t_end = params['t_end']
  t_step = params['t_step']
  t_win = params['t_win']
  t_chunk = params['t_chunk']
  include = params['include']
  exclude = params['exclude']

  ct = 1; good = 0; total = 0
  P = {} # site_id's --> position estimate
  C = {} # site_id's --> confidence region estimate
  for t in np.arange(t_start, t_end+t_chunk, t_chunk):
      

    # Signal data
    sig = signal.Signal(db_con, dep_id, t, t+t_chunk, include=include, exclude=exclude,
                          score_threshold=0.15)
    print "chunk", ct, '(%d pulses)' % sig.get_count()
    ct += 1
    if sig.t_start == float("+inf"):
      continue

    # Compute positions
    positions = position.WindowedPositionEstimator(sig, sites, site34, sv, 
                             t_step, t_win, method=signal.Signal.Bartlet)
   
    for pos in positions:

      print pos.splines.keys(), 
      site_ids = tuple(set(pos.splines.keys()))
      if P.get(site_ids) is None:
        P[site_ids] = []
        C[site_ids] = []
      
      P[site_ids].append(pos.p)
      
      if pos.p is not None:
        try: 
          cov = position.BootstrapCovariance(pos, sites, max_resamples=500)
          E = cov.conf(conf_level)
          C[site_ids].append((E.angle, E.axes[0], E.axes[1]))
          print "Ok"
          good += 1
        except position.SingularError:
          C[site_ids].append(None)
          print "non positive indefinite"
        except position.PosDefError: 
          C[site_ids].append(None)
          print "non positive indefinite (PosDefError)"
        except position.BootstrapError:
          C[site_ids].append(None)
          print "samples ..." 
      else: C[site_ids].append(None)
      total += 1

  print 'good', good, "out of", total, "(t_win=%d, norm=%s)" % (
                              t_win, position.NORMALIZE_SPECTRUM)
  return (P, C)



def plot_map(p_known, sites, fn): 
  fig = pp.gcf()
  ax = fig.add_subplot(111)
  ax.axis('equal')
  ax.set_xlabel('easting (m)')
  ax.set_ylabel('northing (m)')
  offset = 20
  for (id, p) in sites.iteritems():
    pp.plot(p.imag, p.real, color='r', marker='o', ms=7)
    pp.text(p.imag+offset, p.real+offset, id)
  pp.plot(p_known.imag, p_known.real, color='w', marker='o', ms=7)
  pp.grid()
  pp.title("Receivers and transmitter")
  pp.savefig("%s.png" % (fn), dpi=120)
  pp.clf()
  

def mean(P, site_ids): 
  pos = np.array(filter(lambda x : x != None, P[site_ids]))
  pos = pos[~np.isnan(pos)]
  if len(pos) > 0: 
    return np.mean(pos)
  else:
    return None

def plot(P, fn, p_known=None):
  for site_ids in P.keys():
    p_mean = mean(P, site_ids)
    if len(site_ids) > 1:
      pos = np.array(filter(lambda p: p!=None, P[site_ids]))
      #pos = pos[np.abs(pos - p_known) < 200]
      fig = pp.gcf()
      ax = fig.add_subplot(111)
      ax.axis('equal')
      ax.set_xlabel('easting (m)')
      ax.set_ylabel('northing (m)')
      pp.scatter(np.imag(pos), np.real(pos), alpha=0.3, facecolors='b', edgecolors='none', s=25)#, zorder=11)
      if p_known is not None: 
        pp.plot(p_known.imag, p_known.real, color='w', marker='o', ms=7)
      else: 
        pp.plot(p_mean.imag, p_mean.real, color='w', marker='o', ms=7)
      pp.grid()
      pp.title("sites=%s total estimates=%d" % (str(site_ids), len(pos)))
      pp.tight_layout()
      pp.savefig("%s-%s.png" % (fn, ''.join(map(lambda x: str(x), site_ids))), dpi=120, bbox_inches='tight')
      pp.clf()

def count(C): 
  res = {}
  for site_ids in C.keys():
    res[site_ids] = len(filter(lambda ell: ell!=None, C[site_ids]))
  return res

def correlation(P, C, p_known=None): 
  res = {}
  for site_ids in P.keys():
    val = []; dist = []
    if len(site_ids) > 1 and len(P[site_ids]) >= 10:
      p_mean = mean(P, site_ids)
      for i in range(len(P[site_ids])): 
        if C[site_ids][i] is not None and C[site_ids][i][1] > 0 and C[site_ids][i][2]> 0: 
          angle, axis0, axis1 = C[site_ids][i]
          E = position.Ellipse(P[site_ids][i], angle, [axis0, axis1])
          val.append(E.area())
          if p_known is None: 
            dist.append(np.abs(P[site_ids][i] - p_mean))
          else:
            dist.append(np.abs(P[site_ids][i] - p_known))
      # First value is the correlation, the second is the p-value 
      # (probability of data asuuming they are uncorrelated)
      res[site_ids] = scipy.stats.stats.pearsonr(val, dist)
    else: res[site_ids] = None
  return res

def coverage(P, C, p_known=None): 
  res = {}
  for site_ids in P.keys():
    val = []
    if len(site_ids) > 1:
      p_mean = mean(P, site_ids)
      for i in range(len(P[site_ids])): 
        if C[site_ids][i] is not None and C[site_ids][i][1] > 0 and C[site_ids][i][2]> 0: 
          angle, axis0, axis1 = C[site_ids][i]
          E = position.Ellipse(P[site_ids][i], angle, [axis0, axis1])
          if p_known is None:
            val.append(p_mean in E)
          else: val.append(p_known in E)
      if len(val) > 0: 
        res[site_ids] = (sum(val), len(val))
      else: 
        res[site_ids] = None
    else: res[site_ids] = None
  return res




if __name__ == '__main__':  
  db_con = util.get_db('reader')

  cal_id = 3
  site34 = np.complex(4260910.87, 574296.45)
  
  sv = signal.SteeringVectors(db_con, cal_id)
  sites = util.get_sites(db_con)
  (center, zone) = util.get_center(db_con)
  
  # Becaon parameters
  params = { 't_step' : 30, 
             't_win' : 60, 
             't_chunk' : 3600 / 4,
             'dep_id' : 60,
             't_start' : 1383098400.51432,
             't_end' : 1383443999.351099,
             'include' : [],
             'exclude' : [] } 
  
  # Mary-Brook-walk-around params
  #dep_id = 61
  #t_start = 1396725598.548015
  #t_end = 1396732325.777558

  prefix = 'f30beacon' 
  fn = prefix + ''.join(map(lambda id: str(id), params['include']))
  if len(params['exclude']) > 0:
    fn += '.e' + ''.join(map(lambda id: str(id), params['exclude']))
  print fn
     
  #band_dist(params['dep_id'], sites, db_con)
  #plot_map(site34, sites, 'beacon-map')
  
  #P, C = process(sv, sites, center, params)
  #pickle.dump((P, C), open(fn+'.'+suffix, 'w'))
  (P, C) = pickle.load(open(fn+'.'+suffix, 'r'))

  plot(P, fn, site34)

  a = sorted(P.keys())

  print "Count"
  ct = count(C)
  for site_ids in a:
    print site_ids, '-->', ct[site_ids]

  print "\nCorrelation" # of distance to true position and ellipse area
  corr = correlation(P, C, site34)
  for site_ids in a:
    if corr[site_ids] == None:
      print site_ids
    else:
      print site_ids, '--> %0.4f, p-val=%0.4f' % corr[site_ids]

  print "\nCvg. probability" 
  cover = coverage(P, C)
  for site_ids in a:
    p = cover[site_ids]
    if p == None:
      print site_ids
    elif p[0] == 0:
      print site_ids, '%d/%d' % (p[0], p[1])
    else:
      print site_ids, '--> %%%.1f %d/%d' % ((float(100*p[0])/p[1],) + p)
 

