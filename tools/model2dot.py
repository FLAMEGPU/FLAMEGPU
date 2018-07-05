#!/usr/bin/python
# $Id$
#
# Authors: Mozhgan Kabiri Chimeh, Paul Richmond (based on Alcione de Paiva Oliveira existing version)
#
# This program is an extended/modified version written previously by Alcione de Paiva Oliveira.
# Any questions, please contact m.kabiri-chimeh@sheffiled.ac.uk or p.richmond@sheffield.ac.uk
#
# Date : July 2017
#
# Description:
# Create a dot direct graph from the Flame GPU model file.
# By default the nodes are renamed to avoid direct cycles, so improving readability.
#
# Note: Feel free to remove "splines=ortho;", then you will have curved lines
#---------------------------------------------------------------------
#
# input arguments
# Example python3 model2dot.py -i XMLModelFile.xml -o out.gdot

# To convert the dot graph to png: dot -Tpng out.gdot -o out.png


import getopt, sys

from xml.dom.minidom import parse
import xml.dom.minidom

funcOrder =[]
previousState =[]
afterState=[]
conditions=[]
cycle = False
genFunc = False
entrada = ""
saida = ""
mes = []
primeiro = True
clusterNumber = 0

#---------------------------------------------------------------------
#    Get node in text format
#---------------------------------------------------------------------
def getNodeText(node):
    nodelist = node.childNodes
    result = []
    for node in nodelist:
        if node.nodeType == node.TEXT_NODE:
            result.append(node.data)
    return ''.join(result)

#---------------------------------------------------------------------
#    Get condition in text format
#---------------------------------------------------------------------
def getCondition(node):
    globalCond = False
    condT = (node.getElementsByTagName('condition'))
    if (not condT):
       condT = (node.getElementsByTagName("gpu:globalCondition"))
       if (not condT):
           return ""
       else:
           globalCond = True

    lhsNode = node.getElementsByTagName('lhs')[0].getElementsByTagName('agentVariable')
    if (lhsNode):
        lhs = getNodeText(lhsNode[0])
    else:
        lhs = getNodeText(node.getElementsByTagName('lhs')[0].getElementsByTagName('value')[0])

    rhsNode = node.getElementsByTagName('rhs')[0].getElementsByTagName('agentVariable')
    if (rhsNode):
        rhs = getNodeText(rhsNode[0])
    else:
        rhs = getNodeText(node.getElementsByTagName('rhs')[0].getElementsByTagName('value')[0])

    operator = getNodeText(node.getElementsByTagName('operator')[0])
    if (operator == "&lt;"):
        operator = "<"
    elif (operator == "&le;"):
        operator = "<="
    elif (operator == "&gt;"):
        operator = ">"
    elif (operator == "&ge;"):
        operator = ">="

    if (globalCond):
        return "[ label =\"global :"+ lhs+operator+rhs+"\"]"
    return "[ label =\""+ lhs+operator+rhs+"\"]"


#---------------------------------------------------------------------
#    Rename states ocurring more than once
#---------------------------------------------------------------------
def renameStates(agent):
    tam = len(funcOrder)
    for i in range(tam):
       if(previousState[i].find(agent+":")==0 and previousState[i]==afterState[i]):
          afterState[i] = afterState[i]+"\'"
          j = i+ 1
          while (j < tam):
             if (previousState[i] == previousState[j]):
                previousState[j] = previousState[j]+"\'"
             if (previousState[i] == afterState[j]):
                afterState[j] = afterState[j]+"\'"
             if (afterState[i] == afterState[j]):
                afterState[j] = afterState[j]+"\'"
             j = j+ 1


# input arguments
# Example python3 model2dot.py -i XMLModelFile.xml -o out.gdot -f -c

try:
    opts, args = getopt.getopt(sys.argv[1:], "i:o:fc", ["in=", "out="])
except getopt.GetoptError:
    print ('model2dot.py [-f] [-c] -i <model> -o <output>')
    print ('  -c : can have direct cycles')
    print ('  -f : outputs a source code skeleton of the functions used on the Model')
    sys.exit(' ')
for o, a in opts:
    if o in ("-i", "--in"):
        entrada = str(a)
    if o in ("-o", "--out"):
        saida = str(a)
    if o in ("-f"):
        genFunc = True
    if o in ("-c"):
        cycle = True

#-----------------------------------------------------------
#  If is to output source code in C, writes the preamble
#-----------------------------------------------------------
if (len(sys.argv)==4 and sys.argv[3]=="-f" ):
   genFunc = True
   ffunc = open("functions.cc", "w")
   ffunc.write("#ifndef _FUNCTIONS_H_\n#define _FUNCTIONS_H_\n\n#include \"header.h\"\n\n")

# Open XML document using minidom parser
DOMTree = xml.dom.minidom.parse(entrada)
collection = DOMTree.documentElement

file = open(saida, "w")
file.write("digraph model {\n")

####################

init_flag = False
step_flag = False
exit_flag = False

file.write("  newrank=true;\ncompound=true; \n splines=ortho;\n")

file.write("  START [style=invisible];\n");
file.write("  MID [style=invisible];\n");
file.write("  END [style=invisible];\n");

initFunc = collection.getElementsByTagName("gpu:initFunction")
stepFunc = collection.getElementsByTagName("gpu:stepFunction")
exitFunc = collection.getElementsByTagName("gpu:exitFunction")


name = ""

# init functions
file.write("subgraph cluster_%d{\n color=blue; label=initFunctions;penwidth=3;  \n\n" % (clusterNumber))
clusterNumber = clusterNumber +1

for fi in initFunc:
  name = (fi.getElementsByTagName('gpu:name')[0]).childNodes[0].data
  file.write("   %s [shape=box,penwidth=3];\n" %  name )
  print ("init function: %s" % name)
  init_flag=True
file.write("}\n\n")

init_name_tmp = name
init_temp=clusterNumber-1
#file.write("%s -> START[style=invis];\n"% name)


# step functions
file.write("subgraph cluster_%d{\n color=blue;label=stepFunctions;penwidth=3; \n\n" % (clusterNumber))
clusterNumber = clusterNumber +1

for fs in stepFunc:
   name = (fs.getElementsByTagName('gpu:name')[0]).childNodes[0].data
   file.write("   %s [shape=box,penwidth=3];\n" %  name )
   print ("step function: %s" % name)
   step_flag=True
file.write("}\n\n")
step_name_tmp = name
step_temp=clusterNumber-1

# exit functions
file.write("subgraph cluster_%d{\n color=blue; label=exitFunctions;penwidth=3; \n\n" % (clusterNumber))
clusterNumber = clusterNumber +1

for fe in exitFunc:
   name = (fe.getElementsByTagName('gpu:name')[0]).childNodes[0].data
   file.write("   %s [shape=box,penwidth=3];\n" %  name )
   print ("exit function: %s" % name)
   exit_flag=True
file.write("}\n\n")
exit_name_tmp = name
exit_temp=clusterNumber-1

# fix, it should replace if not exists , get rid of if conditions
if (init_flag):
  if (step_flag):
    file.write("%s -> %s [ltail=cluster_%d,lhead=cluster_%d];\n\n"% (init_name_tmp,step_name_tmp,init_temp,step_temp))
  elif (exit_flag):
    file.write("%s -> %s [ltail=cluster_%d,lhead=cluster_%d];\n\n"% (init_name_tmp,exit_name_tmp,init_temp,exit_temp))
elif (step_flag & exit_flag):
  file.write("%s -> %s [ltail=cluster_%d,lhead=cluster_%d];\n\n"% (step_name_tmp,exit_name_tmp,step_temp,exit_temp))


#################### Rank function acc to layers



f_Layers = collection.getElementsByTagName("layers")[0]
funcSynch = f_Layers.getElementsByTagName("layer")

file.write("{node [shape=plaintext, fontsize=16];")
file.write("/* the time-line graph */\n Layer")

layer=0
for f in funcSynch:
  layer+=1
  file.write("->%d" % layer)
file.write(";}\n\n")

layer=0  
for f in funcSynch:
  layer+=1
  fl = f.getElementsByTagName("gpu:layerFunction")
  file.write("{rank = same ;%d;"%layer)
  for g in fl:
      nameFunc = (g.getElementsByTagName('name')[0]).childNodes[0].data
      file.write("%s ;"% nameFunc)
  file.write("}\n\n")

####################


# Get all the agents in the collection
agents = collection.getElementsByTagName("gpu:xagent")
layers = collection.getElementsByTagName("gpu:layerFunction")

# Get a list of the function execution in the correct order
for layer in layers:
   nomeFuncLayer = (layer.getElementsByTagName('name')[0]).childNodes[0].data
   funcOrder.append(nomeFuncLayer)
   previousState.append("")
   afterState.append("")
   conditions.append("")
   #print ("Function:%s\n" % (nomeFuncLayer))


# Print detail of each agent.
for agent in agents:
   nome = (agent.getElementsByTagName('name')[0]).childNodes[0].data
   print ("agent: %s" % nome)

   if not primeiro:
      file.write("}\n\n")

   primeiro = False
   file.write("subgraph cluster_%d{\n label=\"%s\";color=blue; penwidth=3; \n\n" % (clusterNumber,nome))
   clusterNumber = clusterNumber +1

   funcs = agent.getElementsByTagName('functions')[0]
   funs = funcs.getElementsByTagName("gpu:function")
   for f in funs:
      noIO =True
      nameFunc = (f.getElementsByTagName('name')[0]).childNodes[0].data
      if (nameFunc not in funcOrder):
         print ('  warning : funtion %s is not in layers list.\n' % (nameFunc))

      print("Function:"+nameFunc)
      condition = getCondition(f)
      rngT = (f.getElementsByTagName('gpu:RNG'))
      if (rngT):
          rng = rngT[0].childNodes[0].data
      else:
          rng = "false"
#      description = f.getElementsByTagName('description')[0]
      ist = (f.getElementsByTagName('currentState')[0]).childNodes[0].data
      fst = (f.getElementsByTagName('nextState')[0]).childNodes[0].data

#      print ("Funcao: %s" % nameFunc)

      # Store the states of each function
      previousState[funcOrder.index(nameFunc)] = nome+":"+ist
      afterState[funcOrder.index(nameFunc)]=nome+":"+fst

      conditions[funcOrder.index(nameFunc)]= condition

      file.write("   %s [shape=box,penwidth=3];\n" %  nameFunc )

      if (cycle):
          file.write("   %s -> %s %s [penwidth=3];\n" % (ist, nameFunc, condition ))
          file.write("   %s -> %s[penwidth=3];\n" %  (nameFunc,  fst))

      outputs = f.getElementsByTagName('outputs')
      if (outputs.length>0):
          gpuOut = outputs[0].getElementsByTagName("gpu:output")[0]
          menName = getNodeText(gpuOut.getElementsByTagName("messageName")[0])


          file.write("   %s -> %s [color=green4,penwidth=3];\n" %  (nameFunc,  menName))


          if (menName not in mes):
              mes.append(menName)



      inputs = f.getElementsByTagName('inputs')
      if (inputs.length>0):
          gpuIN = inputs[0].getElementsByTagName("gpu:input")[0]
          menName = getNodeText(gpuIN.getElementsByTagName("messageName")[0])


          file.write("   %s -> %s [color=green4,penwidth=3];\n" %  ( menName, nameFunc))


          if (menName not in mes):
              mes.append(menName)



    #----------------------------------------------------------------------------------
    #  If is not to produce cycles on the same state then the states will be renamed
    #----------------------------------------------------------------------------------
   if (not cycle):
      renameStates(nome)
      last_state=0
      for k in range(len(funcOrder)):
          if(previousState[k].find(nome+":")==0 ):
             file.write("   \"%s\" -> %s %s;\n" % (previousState[k][previousState[k].find(":")+1:], funcOrder[k], conditions[k] ))
             print("   \"%s\" -> %s %s;\n" % (previousState[k][previousState[k].find(":")+1:], funcOrder[k], conditions[k] ))
             file.write("   %s -> \"%s\";\n" %  (funcOrder[k],  afterState[k][afterState[k].find(":")+1:]))
             print ("   %s -> \"%s\";\n" %  (funcOrder[k],  afterState[k][afterState[k].find(":")+1:]))
             last_state=k
      file.write("  \"%s\"-> MID [style=invis];\n" %  (afterState[last_state][afterState[last_state].find(":")+1:]))

    #  file.write(" START-> \"%s\"[style=invis] ;\n"% previousState[0][previousState[0].find(":")+1:])


#----------------------------------------------------------------------------------
#  Writes the shape of messages
#----------------------------------------------------------------------------------
file.write("}\n\n")


file.write(" START-> \"%s\"[style=invis] ;\n"% previousState[0][previousState[0].find(":")+1:])


file.write("MID -> END [style=invis];\n\n")

if (init_flag):
  file.write("%s -> START [style=invis];\n"% (init_name_tmp))
  file.write("{rank = same ; START ; %s;}\n"% (init_name_tmp))
if (step_flag):
  file.write("{rank = same ; MID ; %s;}\n"% (step_name_tmp))
if (exit_flag):
  file.write("{rank = same ; END ; %s;}\n\n"% (exit_name_tmp))

for m in mes:

    file.write("   %s [shape=box][shape=diamond, label=%s, fontcolor=green4, color=green4,penwidth=3];\n" % (m,m))


file.write("}")

