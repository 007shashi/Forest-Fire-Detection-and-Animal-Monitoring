from detecto import core, utils, visualize
model = core.Model.load('model_weights.pth', ['tiger', 'elephant', 'panda'])
def process(file_path):
    image = utils.read_image(file_path)
    predictions = model.predict(image)

    labels, boxes, scores = predictions

    scores=scores

    alt_score=[]
    for i in scores:
        alt_score.append(float(i))

    ele=[0]
    tig=[0]
    pan=[0]
    j=0
    for i in labels:
        if i=="elephant":
            ele.append(alt_score[j])
        elif i=="tiger":
            tig.append(alt_score[j])
        elif i=="panda":
            pan.append(alt_score[j])
        j=j+1
    final=[]    
    elephant_score=max(ele)
    tiger_score=max(tig)
    panda_score=max(pan)
    elephant_score=round(elephant_score*100,2)
    tiger_score=round(tiger_score*100,2)
    panda_score=round(panda_score*100,2)
    if (elephant_score>75):
        final.append("Elephant")
    if(tiger_score>75):
        final.append("Tiger")
    if(panda_score>75):
        final.append("Panda")
    print("Result==",final)
    prob=0.0
    if final[0]=="Elephant":
        prob=elephant_score
    if final[0]=="Tiger":
        prob=tiger_score
    if final[0]=="Panda":
        prob=panda_score
    return final[0],prob
#print(process("./elephant.jpg"))
    
        
        
