#python Exp.py --graph_type=Small-World --numExp=30
#python Exp.py --graph_type=BA --numExp=30
#python Exp.py --graph_type=BTER --numExp=30
#python Exp.py --graph_type=Email --numExp=1
#python Exp.py --graph_type=Stoc-Block --numExp=30


#python sis_simulations.py --graph_type=BA --numExp=30 --location=random
#python sis_simulations.py --graph_type=Small-World --numExp=30 --location=random
#python sis_simulations.py --graph_type=BTER --numExp=30 --location=random
#python sis_simulations.py --graph_type=Email --numExp=1 


  #graph_type=$1
  #numExp=$2
  #weighted=$3
  #save=$4
  #python main.py --graph_type=$graph_type --mode=721 --numExp=$numExp --weighted=$weighted --save_result=$save
  #python main.py --graph_type=$graph_type --mode=equalAlpha --numExp=$numExp --weighted=$weighted --save_result=$save
  #python main.py --graph_type=$graph_type --mode=alpha1=1   --numExp=$numExp --weighted=$weighted --save_result=$save
  #python main.py --graph_type=$graph_type --mode=alpha2=0   --numExp=$numExp --weighted=$weighted --save_result=$save
  #python main.py --graph_type=$graph_type --mode=alpha3=1   --numExp=$numExp --weighted=$weighted --save_result=$save

graph_type=Brain
numExp=1
python sis_simulations.py --graph_type=$graph_type  --numExp=$numExp --gamma=0.24 --tau=0.2 --weighted=weighted --algo=ours
python sis_simulations.py --graph_type=$graph_type  --numExp=$numExp --gamma=0.24 --tau=0.2 --weighted=weighted --algo=deg
python sis_simulations.py --graph_type=$graph_type  --numExp=$numExp --gamma=0.24 --tau=0.2 --weighted=weighted --algo=gel


#graph_type=Small-World
#numExp=30
#python sis_simulations.py --graph_type=$graph_type  --numExp=$numExp

