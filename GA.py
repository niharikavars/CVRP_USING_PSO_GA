#Importing required modules
import numpy 
import math 
import random as r
import copy
import matplotlib.pyplot as plt
import PlotTours

def function_obj3(Cluster, DisM):
    CurDis = 0

    for i in range(len(Cluster)):
        if i == 0: #represents from depot to customer
            CurDis = CurDis + DisM[0][ Cluster[i] ]
        else:
            CurDis = CurDis + DisM[ Cluster[i-1] ][ Cluster[i] ]
    #represents from customer to depot
    CurDis = CurDis + DisM[ Cluster[len(Cluster)-1] ][0]
    return CurDis

#function_obj2(Customers, DistanceMatrix, CurrentSolution, CurrentSubRouteSum)
#This function computes total travelled distance by each vehicle and route balance
def function_obj2(C, DisM, CS, CSRS):
    ReqVeh = len(CSRS)
    VehDistance = [0]*ReqVeh
    LB=0
    UB=0
    clusterList = []
    for j in range(ReqVeh):
        LB = sum( CSRS[0:j] )
        UB = LB + CSRS[j]
        VehDistance[j] = function_obj3(CS[LB:UB], DisM)
        
    return sum(VehDistance), (max(VehDistance)-min(VehDistance))

#function_obj1(Customers, VehicleCapacity, DemandMatrix, CurrentSolution)
#This function computes number of customers visited by each vehicle
def function_obj1(C, VCap, DM, CS): 
    CusServe = [ ]
    C = len(CS)

    Flag = True
    while Flag == True:
        CurCap = 0
        CusServeV = 0
        k=sum(CusServe)
        for i in range(k, C):
            if CurCap + DM[ CS[i] ] > VCap:
                break
            else:
                CusServeV = CusServeV + 1
                CurCap  = CurCap + DM[ CS[i]  ]
        CusServe.append(CusServeV)

        if sum(CusServe) == C :
            Flag = False
            
    #print(len(CusServe), CusServe);input()
    return CusServe

#Function to calculate the objective values
#function_obj(Customers, DemandMatrix, DistanceMatrix, Vehicles, VehicleCapacity, solutions)
def function_obj(C, DM, DisM, VCap, sol):
    pop_size = len(sol)

    #print(sol, pop_size);input() 
    Obj1_value = [0]*pop_size #Minimum required vehicles
    Obj2_value = [0]*pop_size #Total travelled Distance-The total distance travelled by all vehicles
    Obj3_value = [0]*pop_size #Route Imbalance-The cost difference b/t most expensive and least expensive tour

    #print(Vehicles, Customers, pop_size)
    for i in range(pop_size):
        CurrentSolution = sol[i]
        CurrentSubRouteSum = function_obj1(C, VCap, DM, CurrentSolution)
        Obj1_value[i] = len(CurrentSubRouteSum) #Minimum required vehicles
        Obj2_value[i], Obj3_value[i] = function_obj2(C, DisM, CurrentSolution, CurrentSubRouteSum)

    return Obj1_value, Obj2_value, Obj3_value

#Function to sort the chromosomes 
def sort_list_Angle(list1, list2):
    zipped_pairs = zip(list2, list1)
    z = [x for _, x in sorted(zipped_pairs)] 
    return z

#Function to carry out the mutation operator
def mutation(solution):
    r1=r.randint(0, len(solution)-1)
    r2=r.randint(0, len(solution)-1)
    solution[r1], solution[r2] = solution[r2], solution[r1]
    return solution

#Function to carry out the crossover
def crossoverSRC(sol1, sol2, Customers, VehCap, DisM, DemMat):
    tempSol1 = []
    tempSol2 = []
    solLength = len(sol1)

    AverageDist = [0]*(Customers+1)
    for i in range(1, Customers+1):
        AverageDist[i] = (sum( DisM[i] ) - DisM[i][0] )/ ( Customers-1 )

    CRSI = 0
    for i in range(0, solLength):
        if len(tempSol1) == 0:
            tempSol1.append(r.randint( 1, Customers ))
        else:
            currentCust = tempSol1[ len(tempSol1)-1 ]
            nextCustSol1Index = sol1.index( currentCust )
            nextCustSol2Index = sol2.index( currentCust )

            if nextCustSol1Index == solLength - 1:
                nextCustSol1 = sol1[0]
            else:
                nextCustSol1 = sol1[nextCustSol1Index +1]

            if nextCustSol2Index == solLength - 1:
                nextCustSol2 = sol2[0]
            else:
                nextCustSol2 = sol2[nextCustSol2Index +1]

            dist1 = DisM[currentCust][nextCustSol1]
            dist2 = DisM[currentCust][nextCustSol2]
            if dist1 <= dist2 and dist1 <= AverageDist[currentCust] and nextCustSol1 not in tempSol1 :
                tempSol1.append( nextCustSol1 )
            elif dist2 <= dist1 and dist2 <= AverageDist[currentCust] and nextCustSol2 not in tempSol1 :
                tempSol1.append( nextCustSol2 )
            else:
                MinDist = 444444444
                for k in range(1, Customers+1):
                    if k not in tempSol1 and DisM[currentCust][k] < MinDist:
                        MinDist = DisM[currentCust][k] 
                        nextCust = k
                tempSol1.append(nextCust)

        CRD = [ DemMat[ tempSol1[l] ] for l in range(CRSI, len(tempSol1))]
        if sum(CRD) > VehCap:
            tempSol1.pop()
            CurCust = tempSol1[len(tempSol1) -1]
            MinDist1 = 444444444
            for k in range(1, Customers+1):
                if k not in tempSol1 and DisM[CurCust][k] < MinDist1:
                    MinDist1 = DisM[CurCust][k]
                    nextCust1 = k
            newCRD = [ DemMat[ tempSol1[l] ] for l in range(CRSI, len(tempSol1))]
            if sum(newCRD) + DemMat[nextCust1] <=  VehCap:
                tempSol1.append(nextCust1)
            else:
                CRSI = len( tempSol1)
                Flag = True
                while Flag == True:
                    nextCust = r.randint( 1, Customers )
                    if nextCust not in tempSol1:
                        tempSol1.append(nextCust)
                        Flag = False

    CRSI = 0
    for i in range(0, solLength):
        if len(tempSol2) == 0:
            tempSol2.append( r.randint( 1, Customers ) )
        else:
            currentCust = tempSol2[ len(tempSol2)-1 ]
            nextCustSol1Index = sol1.index( currentCust )
            nextCustSol2Index = sol2.index( currentCust )

            if nextCustSol1Index == solLength - 1:
                nextCustSol1 = sol1[0]
            else:
                nextCustSol1 = sol1[nextCustSol1Index +1]

            if nextCustSol2Index == solLength - 1:
                nextCustSol2 = sol2[0]
            else:
                nextCustSol2 = sol2[nextCustSol2Index +1]

            dist1 = DisM[currentCust][nextCustSol1]
            dist2 = DisM[currentCust][nextCustSol2]
            if dist1 <= dist2 and dist1 <= AverageDist[currentCust] and nextCustSol1 not in tempSol2 :
                tempSol2.append( nextCustSol1 )
            elif dist2 <= dist1 and dist2 <= AverageDist[currentCust] and nextCustSol2 not in tempSol2:
                tempSol2.append( nextCustSol2 )
            else:
                MinDist = 444444444
                for k in range(1, Customers+1):
                    if k not in tempSol2 and DisM[currentCust][k] < MinDist:
                        MinDist = DisM[currentCust][k] 
                        nextCust = k
                tempSol2.append(nextCust)

        CRD = [ DemMat[ tempSol2[l] ] for l in range(CRSI, len(tempSol2))]
        if sum(CRD) > VehCap:
            tempSol2.pop()
            CurCust = tempSol2[len(tempSol2) -1]
            MinDist1 = 444444444
            for k in range(1, Customers+1):
                if k not in tempSol2 and DisM[CurCust][k] < MinDist1:
                    MinDist1 = DisM[CurCust][k]
                    nextCust1 = k
            newCRD = [ DemMat[ tempSol2[l] ] for l in range(CRSI, len(tempSol2))]
            if sum(newCRD) + DemMat[nextCust1] <=  VehCap:
                tempSol2.append(nextCust1)
            else:
                CRSI = len( tempSol1)
                Flag = True
                while Flag == True:
                    nextCust = r.randint( 1, Customers )
                    if nextCust not in tempSol2:
                        tempSol2.append(nextCust)
                        Flag = False

    #print(tempSol1, len(tempSol1))
    #print(tempSol2, len(tempSol1)); input()
    CSRS1 = function_obj1(Customers, VehCap, DemMat, tempSol1)
    ReqVeh = len(CSRS1)
    LB=0;UB=0
    newSol1 = []
    for j in range(0, ReqVeh):
        LB = sum( CSRS1[0:j] ); UB = LB + CSRS1[j]
        CurRoute = tempSol1[LB:UB]
        newRoute = []
        lenRoute = len(CurRoute)
        for i in range(0, lenRoute): 
            if i == 0:
                CurCust = 0
            else:
                CurCust = newRoute[ len(newRoute) -1]
                
            MinDist = 444444444
            for k in CurRoute:
                if k not in newRoute and DisM[CurCust][k] < MinDist:
                    MinDist = DisM[CurCust][k]
                    nextCust = k
            newRoute.append(nextCust)
        newSol1.extend(newRoute)
    tempSol1 = copy.deepcopy( newSol1)   

    CSRS2 = function_obj1(Customers, VehCap, DemMat, tempSol2)
    ReqVeh = len(CSRS2)
    LB=0;UB=0
    newSol2 = []
    for j in range(0, ReqVeh):
        LB = sum( CSRS2[0:j] ); UB = LB + CSRS2[j]
        CurRoute = tempSol2[LB:UB]
        newRoute = []
        lenRoute = len(CurRoute)
        for i in range(0, lenRoute): 
            if i == 0:
                CurCust = 0
            else:
                CurCust = newRoute[ len(newRoute) -1]
                
            MinDist = 444444444
            for k in CurRoute:
                if k not in newRoute and DisM[CurCust][k] < MinDist:
                    MinDist = DisM[CurCust][k]
                    nextCust = k
            newRoute.append(nextCust)
        newSol2.extend(newRoute)

    tempSol2 = copy.deepcopy( newSol2)   


    return tempSol1, tempSol2

#Tournament Selection
def selection(function2_values, function3_values):
    NumberSol = len(function2_values)
    n1 = r.randint(0, NumberSol-1)
    n2 = r.randint(0, NumberSol-1)
    n3 = r.randint(0, NumberSol-1)
    n4 = r.randint(0, NumberSol-1)
    n5 = r.randint(0, NumberSol-1)
    n6 = r.randint(0, NumberSol-1)
    n7 = r.randint(0, NumberSol-1)
    n8 = r.randint(0, NumberSol-1)
    
    tempSolNum1=0
    tempSolNum2=0
    tempSolNum3=0
    tempSolNum4=0
    if function2_values[n1] < function2_values[n2]:
        tempSolNum1 = n1
    else:
        tempSolNum1 = n2

    if function2_values[n3] < function2_values[n4]:
        tempSolNum2 = n3
    else:
        tempSolNum2 = n4

    if function2_values[n5] < function2_values[n6]:
        tempSolNum3 = n5
    else:
        tempSolNum3 = n6

    if function2_values[n7] < function2_values[n8]:
        tempSolNum4 = n7
    else:
        tempSolNum4 = n8


    SolNum1=0
    SolNum2=0
    if function2_values[tempSolNum1] < function2_values[tempSolNum2]:
        SolNum1 = tempSolNum1
    else:
        SolNum1 = tempSolNum2

    if function2_values[tempSolNum3] < function2_values[tempSolNum4]:
        SolNum2 = tempSolNum3
    else:
        SolNum2 = tempSolNum4

    tempSolNum = 0 
    if function2_values[SolNum1] < function2_values[SolNum2]:
        tempSolNum = SolNum1
    elif function2_values[SolNum2] < function2_values[SolNum1]:
        tempSolNum = SolNum2
    else:
        if function3_values[SolNum1] >= function3_values[SolNum2]:
            tempSolNum = SolNum1
        else:
            tempSolNum = SolNum2
            
    return tempSolNum

#GA(Customers, DemMat, DistMat, VehCap, solutions, Num_Iterations)
def GA(C, DM, DisM, VCap, solution, Num_Iterations):
    pop_size = len(solution)
    max_gen = Num_Iterations
    gen_no=0
    DistancePerGeneration = []
    while(gen_no < max_gen):
        function1_values, function2_values, function3_values = function_obj(C, DM, DisM, VCap, solution)
        IndexWithMinimumValue = function2_values.index( min(function2_values) )
        BestSolution = solution[IndexWithMinimumValue]
        #print(min(function2_values), BestSolution)

        #Generating offsprings
        solution2 = solution[:]
        while( len(solution2) != 2*pop_size ):
            a1  = selection(function2_values, function3_values)
            b1  = selection(function2_values, function3_values)
            while( b1 == a1 ):
                b1 = selection(function2_values, function3_values)
            #print("Beore crossover",a1, b1);input()
            tempSol1, tempSol2 = crossoverSRC(solution[a1],solution[b1],C, VCap, DisM, DM)
            #print(tempSol1, tempSol2)
            mutTempSol1 = mutation(tempSol1)
            mutTempSol2 = mutation(tempSol2)
            tempV1, tempV2, tempV3 = function_obj(C, DM, DisM, VCap, [tempSol1, mutTempSol1, tempSol2, mutTempSol2])
            if tempV2[0] <= tempV2[1]:
                solution2.append(tempSol1)
            else:
                solution2.append(mutTempSol1)

            if tempV2[2] <= tempV2[3]:
                solution2.append(tempSol2)
            else:
                solution2.append(mutTempSol2)
        
        #print("After new population", len(solution2), len(solution))
        function1_values2, function2_values2, function3_values2 = function_obj(C, DM, DisM, VCap, solution2)
        #print("Min values", min(function1_values2), min(function2_values2), min(function3_values2)) 
        #print("Max values", max(function1_values2), max(function2_values2), max(function3_values2))
        #input()
        LengthNewSol=len(solution2)
        list1 = list(range(1, LengthNewSol+1))
        #print(list1)
        #print(function2_values2);input()
        Sorted_List = sort_list_Angle(list1, function2_values2)
        #print(Sorted_List)
        LengthNewSol = len(solution)
        New_Sorted_List = Sorted_List[0:LengthNewSol]
        #print(New_Sorted_List);input()

        
        solution= [solution2[i-1] for i in New_Sorted_List]

        print("Generation Number", gen_no)
        print("Total Travelled Distance", min(function2_values2))
        DistancePerGeneration.append(min(function2_values2))
        gen_no = gen_no + 1

    function1_values, function2_values, function3_values = function_obj(C, DM, DisM, VCap, solution)
    #print("function1_values", function1_values);
    #print("function2_values", function2_values)
    return solution, DistancePerGeneration


#********************  Function to create solutions********************
def CreateSolution(Customers,  Nsolutions):
    solutions = [0 for i in range(Nsolutions)]
    for i in range(0, Nsolutions):
        solutions[i] = copy.deepcopy( list( numpy.random.permutation( range(1, Customers+1) ) ) )
    return solutions

#********************  Function to compute Eucledian distance B/t two points********************
def DistanceFunction(x1, y1, x2, y2):
    return math.sqrt( (x2-x1)**2 + (y2-y1)**2 )

#********************Function to read input data ********************
#Reading files from Set A Instances
def ReadSetA(fileName):
    A = open(fileName,'r')
    B = A.read( )
    A.close()
    B = B.split('\n')
    #print(len(B));input( )

    #Reading number of customers and vehicles; capacity of vehicles
    tempString = B[3].split()
    Customers = int(tempString[2]) # Customers + Depot
    tempString = B[5].split()
    VehicleCapacity = int(tempString[2])

    Rows = Customers
    DemandMatrix = [0 for i in range(Rows)]
    X_Cor = [0 for i in range(Customers)]
    Y_Cor = [0 for i in range(Customers)]
    for i in range(0, Rows):
        tempString1 = B[i+7].split()
        DemandMatrix[i] = int(float( B[Customers + 8 + i].split()[1] ))
        X_Cor[i] = int(float(tempString1[1]))
        Y_Cor[i] = int(float(tempString1[2]))

    #Computing Distance matrix : The distance between any pair of customers and depot
    #Computing Demand matrix : The demand of each customer 
    DistanceMatrix = [[0 for i in range(Customers)] for j in range(Customers)]
    for i in range( Customers ):
        for j in range( Customers ):
            if i == j :
                DistanceMatrix[i][j] = 0
            elif i > j:
                DistanceMatrix[i][j] = DistanceMatrix[j][i]
            else:
                DistanceMatrix[i][j] = DistanceFunction( X_Cor[i], Y_Cor[i], X_Cor[j], Y_Cor[j] )

    return VehicleCapacity, Customers-1, DistanceMatrix, DemandMatrix, X_Cor, Y_Cor


##********************************Main program begins here******************************
#Reading Set A files 
VehCap, Customers, DistMat, DemMat, X_Cor, Y_Cor = ReadSetA('A-n80-k10.vrp')
#print("Customers",Customers, len(X_Cor), len(Y_Cor), X_Cor, Y_Cor);input()
#print("Demand",DemMat, "VehicleCapacity", VehCap);input()
#print("Distance Matrix",DistMat) ;input()

Geometric_Points = [[X_Cor[i], Y_Cor[i]] for i in range(len(X_Cor))]
#print("2D Geometric_Points", Geometric_Points);input()
#angleCusDep = AngleFunction( Geometric_Points )

#Creating Num_Solutions-1 random solutions(Permutations)
Num_Solutions =int(input("How many solutions you are interested to create"))
solutions = CreateSolution(Customers, Num_Solutions)
#print(len(solutions[0]), len(solutions))
#print(solutions)
#tempSol = solutions[0]
#CurrentSubRouteSum = function_obj1( Customers, VehCap, DemMat, tempSol)
#ReqVeh = Vehicles-CurrentSubRouteSum.count(0)
#print(CurrentSubRouteSum,ReqVeh)

#Cluster = []
#for j in range(ReqVeh):
#    LB = sum( CurrentSubRouteSum[0:j] )
 #   UB = LB + CurrentSubRouteSum[j]
  #  Cluster.append(tempSol[LB:UB])
#print(Cluster);input()
#PlotTours.TourFunction(Cluster, Geometric_Points);input()



#Calling Genetic Algorithm
Num_Iterations =int(input("How many generations are required?"))
solutions, DistancePerGeneration = GA(Customers, DemMat, DistMat, VehCap, solutions, Num_Iterations)

function1_values, function2_values, function3_values = function_obj(Customers, DemMat, DistMat, VehCap, solutions)
IndexWithMinValue = function2_values.index(min(function2_values))
BestSolution = solutions[IndexWithMinValue]
print(IndexWithMinValue, function2_values[IndexWithMinValue], BestSolution)
CurrentSubRouteSum = function_obj1( Customers, VehCap, DemMat, BestSolution)
ReqVeh = len(CurrentSubRouteSum)

Cluster = []
for j in range(ReqVeh):
    LB = sum( CurrentSubRouteSum[0:j] )
    UB = LB + CurrentSubRouteSum[j]
    tempCluster = BestSolution[LB:UB]
    Cluster.append( tempCluster)
print("The best solution found", BestSolution)
print("Routes of the cluster", Cluster)
PlotTours.TourFunction(Cluster, Geometric_Points)

#Ploting the solutions
#function1 = [i * 1 for i in function1_values]
#function2 = [j * 1 for j in function2_values]
#function3 = [j * 1 for j in function3_values]

#function2_2 = [j * 1 for j in function2_values]
#function3_2 = [j * 2 for j in function3_values]

#plt.xlabel('Total Travelled Distance', fontsize=15)
#plt.ylabel('Load Imbalance', fontsize=15)
#plt.scatter(function2, function3)
#plt.show()
y = list(range(1, Num_Iterations+1))
plt.ylabel('Total Travelled Distance', fontsize=15)
plt.xlabel('Generation Number', fontsize=15)
plt.plot(y, DistancePerGeneration)
plt.show()
