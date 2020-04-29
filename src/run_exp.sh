#python Exp.py --graph_type=Small-World --numExp=30
#python Exp.py --graph_type=BA --numExp=30
#python Exp.py --graph_type=BTER --numExp=30
#python Exp.py --graph_type=Email --numExp=1
#python Exp.py --graph_type=Stoc-Block --numExp=30


#python sis_simulations.py --graph_type=BA --numExp=30 --location=random
#python sis_simulations.py --graph_type=Small-World --numExp=30 --location=random
#python sis_simulations.py --graph_type=BTER --numExp=30 --location=random
#python sis_simulations.py --graph_type=Email --numExp=1 


graph_type=$1
numExp=$2
#python Exp.py --graph_type=$graph_type --mode=equalAlpha --numExp=$numExp
#python Exp.py --graph_type=$graph_type --mode=alpha1=1   --numExp=$numExp
#python Exp.py --graph_type=$graph_type --mode=alpha2=0   --numExp=$numExp
#python Exp.py --graph_type=$graph_type --mode=alpha3=0   --numExp=$numExp
#python Exp.py --graph_type=$graph_type --mode=alpha3=1   --numExp=$numExp

#graph_type=BA
#numExp=30
python sis_simulations.py --graph_type=$graph_type  --numExp=$numExp --gamma=0.24 --tau=0.2
python sis_simulations.py --graph_type=$graph_type  --numExp=$numExp --gamma=0.5 --tau=0.1
python sis_simulations.py --graph_type=$graph_type  --numExp=$numExp --gamma=0.3 --tau=0.5


#graph_type=Small-World
#numExp=30
#python sis_simulations.py --graph_type=$graph_type  --numExp=$numExp

