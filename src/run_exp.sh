#python Exp.py --graph_type=Small-World --numExp=30
#python Exp.py --graph_type=BA --numExp=30
#python Exp.py --graph_type=BTER --numExp=30
#python Exp.py --graph_type=Email --numExp=1
#python Exp.py --graph_type=Stoc-Block --numExp=30


#python sis_simulations.py --graph_type=BA --numExp=30 --location=random
#python sis_simulations.py --graph_type=Small-World --numExp=30 --location=random
#python sis_simulations.py --graph_type=BTER --numExp=30 --location=random
#python sis_simulations.py --graph_type=Email --numExp=1 


graph_type=Email
numExp=1
python Exp.py --graph_type=$graph_type --mode=equalAlpha --numExp=$numExp
python Exp.py --graph_type=$graph_type --mode=alpha1=1   --numExp=$numExp
python Exp.py --graph_type=$graph_type --mode=alpha2=0   --numExp=$numExp
python Exp.py --graph_type=$graph_type --mode=alpha3=0   --numExp=$numExp
python Exp.py --graph_type=$graph_type --mode=alpha3=1   --numExp=$numExp

#graph_type=BTER
#numExp=30
python sis_simulations.py --graph_type=$graph_type  --numExp=$numExp

#graph_type=Small-World
#numExp=30
#python sis_simulations.py --graph_type=$graph_type  --numExp=$numExp

