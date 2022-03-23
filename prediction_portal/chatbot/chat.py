import random
import json

import torch

from prediction_portal.chatbot.model import NeuralNet
from prediction_portal.chatbot.nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import os
path=os.getcwd()+'\\prediction_portal\\chatbot\\intents.json'
with open(path, 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

FILE = os.getcwd()+"\\prediction_portal\\chatbot\\data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Healthino"

from prediction_portal.chatbot.sentiment_model import predict_custom_tweet_sentiment

response_dict={

     'workplace_anx':
           ["Sad to hear that this happened. However, life has ups and downs and you are not the only person who  might have to face this situation. Instead of wasting time by repenting upon incorrect decisions or bad luck, focus on the future and what positives can be drawn.", "You must try communicating with your colleagues and seniors. Start speaking out your feelings.", "Things will get easy, just have patience. Be persistent with your work and hard work, don't focus on things that are not in your control and soon the problems will disappear.","It is important for you not only to work, but also focus on your mental health.","Don't worry much regarding this, it will all be okay in some time. Besides dedication, one needs to have positive and optimistic outlook towards things he does."],

       'workplace_pos':
           ["Excellent! It seems you have mastered your field','Great to hear this. I think there's more to come','Being a mental health assistant and chatbot, I didn't expect that you would share something positive, but that surely makes me proud of your work and I hope you stay consistent at this level ' ,'Helping different people everyday kind of makes me bored and negative, but now i got something to rejoice!!", "Today it might be your day, but always remember to put a smile on your co-workers' and juniors' faces too! Congratulations!"  ,"Don't forget to appreciate and thank those who contributed and helped you in making this task successful!"],

     'workplace_depr' : ["That's Sad! We all have good, bad, ugly and worse days in our lives. Consider this as one of the worse days and move on. Remember that for being successful, one needs to overcome his own failures and fears.", "Your boss shouldn't have done that. However, consider this as a bad luck. Things always happen for good and you must not be thinking regarding this as a failure, who knows what's kept on the other side.", "You must not start feeling demotivated to depressed, instead share your thoughts with your colleagues and bosses, I am sure, they would respect them.", " one always has to fight for success. Continue your struggle and you might be there one day", "I think you are stressing too much on your work. Remember that work is just a part of your life, not your life. Your body, family and other responsibilities deserve as much attention as your work. Try and prioritise on things. These problems in work would get solved one day or another, but problems and diseases caused by mental health issues can be irreversible."],

     'workplace_sad': [ "Don't worry, we all have bad days. But the positives after all this is that there is always a positive tomorrow that we all look forward to.", "It's the happy things that people always look forward too in sad times, similarly your time would also come where you would be shining as a start in the darkness of the sky ","Don't worry, the work environment would heal gradually. However, it won't be justified to neglect other important things like your health and important family time. One must always keep their professional and personal lives separate. Whatever happened, you must not forget the happiness of your loved ones.", "Try spending your time by practising any hobby, take your family out for a dinner or just let your kids or pets be around you! Time will heal all the pain and you would be fresh again to look forward to your professional goals."],


       'relationship_anx':
           ["Communication is the key for a happy relationship. Tell your partner and show that you love them. Let them know that you understand that fights are affecting their thoughts, feelings, and behavior and that you love them. Reassure them that you are here to support them in their journey to get better.", "Discuss with your partner the different reasons for them to stay. Perhaps it's the dependents, a beloved pet who needs them, or their faith. These reasons can help them hold on a bit longer until the pain subsides.","Give your partner the required space after an argument and let things settle down between the two of you. Concentrate on other activities and take time off each other. "],

       'relationship_pos':
           ["That's wonderful to know that things are working fine for the two of you. Healthino wishes you the best for a great future ahead!", "Only a few relationships are as good as yours. Have a great time together and cheers to all the happy memories.", "I think you guys were destined to be together for your life-time. Healthino wishes to find a relationship just as perfect as yours!", "A healthy relationship is all about balance and chemistry. Healthino identifies true love and wishes that your love lasts forever!!"],

      'relationship_depr': ["Sharing your feelings with friends and family will help you get through this period. Join a support group where you can talk to others in similar situations. Isolating yourself can raise your stress levels, reduce your concentration, and get in the way of your work, other relationships, and overall health.", "Talk about your feelings with your close ones. Knowing that others are aware of your feelings will make you feel less alone with your pain and will help you heal. Writing in a journal can also be a helpful outlet for your feelings." , "Expressing your feelings will liberate you in a way, but it is important not to dwell on the negative feelings or to over-analyze the situation. Getting stuck in hurtful feelings like blame, anger, and resentment will rob you of valuable energy and prevent you from healing and moving forward. Encourage yourself by the fact that new hopes and dreams will eventually replace your old ones.", "Help yourself heal by scheduling daily time for activities you find calming and soothing. Spend time with good friends, go for a walk in nature, listen to music, enjoy a hot bath, get a massage, read a favorite book, take a yoga class, or savor a warm cup of tea." ],
     'relationship_sad': ["Always remember that don't take decisions when you are sad or angry regarding something. Every relationship has its ups and downs. It's only the fight you put that keeps the relationship alive.", "A relationship is never about 'I', it's always about 'us'. Listen to your partner, understand their opinions on the situations and work things out together.", "Let your partner know that you love them, you respect them and you care about them. Communicate your feelings on a regular basis. Healthino believes that love and effort overcome every obstacle!"],


    'family_anx': ["While we would all prefer to get along well with our families, it is not always possible to do so. Generally people rely a lot on families for mental support however in cases like these, you need to talk it through with a friend or a therapist and figure out ways to iron out the differences between you and your family because there is always light at the end of the tunnel.", "It is not always possible to have happy times with our families. There will be times when you face differences and it is completely normal since every person has their own opinion and outlook on life. However what is important is to sort the issues out and bring back the happy moments. If you need someone to share it with, I am always there." , "Sometimes it is important to think from another person's point of view. Our priorities may shift suddenly in a crisis. Make sure you understand and honor the needs of family members or other household residents during the recovery process.", "Almost all of us face tough times with family at least once in our life. In such situations, take time to do something that is meaningful, relaxing and fun to you and your family. Read a book, sit on the porch and enjoy the scenery, enjoy coffee, or have a family movie or game night. These little things will definitely help you get over it."],

'family_pos': ["Happy to hear that you and your family are doing well. Wishing you happy times ahead.", "I am glad that you are in a good mindspace. Family support always helps us stay happy!",  "So happy to know that you and your family are putting a smile on each others' face. There is only better to come.",  "Amazing! I hope you have the best time in the coming days.", "I am extremely happy for you and I hope you have exciting times ahead!" ],

'family_depr': ["Family issues are commonly experienced by many people but intensifying them to an extent where it takes a toll on your mental health is not normal. Either the issue needs to be solved by clearing out things with family or if it is a persistent issue and you know that it won't be solved easily, then opening up about it to a therapist would be better. Take enough sleep and do exercise or yoga to improve your mental state", "Negative people are like a parasite to the soul. They can eat up the willingness to stay happy and when this negativity comes from family, the situation gets worse. As much as we want to be there for our family, it is important you focus on your mental health too. Try doing certain activities alone, away from the source of negativity in your life. Take a walk, go on a trek, try out solo dinners. It will help you heal a little.", "At times when the situation is not in our control, we need to focus on things that are in our control and that is our own self. It is important to become the hero of your story and not let external factors affect your mental health and self confidence. You always have the option to discuss severe problems with a therapist but focusing on your well being is extremely necessary." ],


'family_sad': ["Every family has to go through some ups and downs, it is important to hold on to each other and figure a way out together. I hope you and your family will overcome all the difficulties soon.", "Tough times are unavoidable and unforeseen at times. You need to be strong and keep your loved ones in a good mental space as well. I wish you all the strength to get over this hurdle.", "Just like happy days, we all have sad days too. In the end our biggest support system is our family and tough times will only strengthen your love and the bond that you share.", "As they say, Families are like a fudge, mostly sweet but with a few nuts. I understand the situation is tough for you and your family but I am sure you will get over it together and be there for each other through it all." ],

'friends_sad':[ " I understand. Friends are indeed an important part of a person's life. And that is exactly why you should keep trying to mend the issues and not give up on each other during the ups and downs. Communication, along with having compassion is the key.  ", 
" I get it, having a friend beside you is what gets you through all the bad times. Perhaps, you should try understanding why you/your friend acted like that and identify the root cause in order to fix the issue. Empathy is very important to establish communication and sort things out. ",
"That's sad. However, a friendship with a strong foundation can always be reignited if efforts are taken in the right direction to mend the problems. ",
"Don't worry, a true friend always remembers the good and the bad times spent together and will eventually revive your friendship because it means a lot to them as well."
 ],

'friends_anx': [ " I'm sad to know that and I hope things get better soon, perhaps sending a surprise apology or something that reminds of the great times you have spent together, is a thing that will melt the toughest heart and pave the way for reviving your friendship. ",
"Sorry to hear that. A friend is someone who makes good times better and bad times easier and it can be difficult to not have your go-to person with you. But, it is also true that, if you take that first step towards fixing your friendship, it will surely do some good and you won't have to regret not fighting for your friendship and perhaps, your friend is expecting this effort from you. ",
"Oh, I'm sad to know that. But it's alright, all friendships go through the ups and downs but at the end, if the friendship is deep enough, it will always turn out okay and there's nothing a little friendly chat or a sweet, lovely gift can't fix. "
],

'friends_depr':[ " Oh, that's awful. I understand that it must be hard considering the bond you must have with your friend all this time and that it may look like things aren't going to be alright. But, trust me, time heals everything. Don't lose hope and things will get sorted out soon.",
" That's unfortunate. Perhaps, all you need is a little bit of personal time to distract yourself and get it off your mind. Plus, there are always other people around you who would be just as lovely a friend to you and they can help you out. ",
"I understand, friendships can be difficult. Especially when it's with the person whom you run to when you're sad or facing a problem. But, it's not the end of the World. You can always spend some time with your family and close ones to make yourself feel better and get a much-deserved break from all of it. With time, the hurt will ease and you will feel better.  ",
" It's not easy but things will get better. Perhaps, if it's hurting so much it is time to take a step back for yourself and reevaluate your friendship. After all, you deserve someone who understands you and is equally glad to have you as their friend. Spend some quality alone time pursuing your hobby or listening to your favourite songs. "
],

'friends_pos': [ "That's great. I am glad things are going well because it's important to have a good friend circle around yourself!",
" Glad to hear that! A good friend is what gets you through the tough times and makes your life happier and better in so many ways!",
"A great friendship is something we all need and I am glad you have such nice companions throughout your life's journey."
 "You really are a blessed person to have such great and lovely friends by your side. Hope things will just get better with time!"
]


}

def get_response(msg):
    input_msg=msg
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.5:
        for intent in intents['intents']:
        
            if tag == intent["tag"]:
                print(tag)
                context_tags=['family','relationship', 'friends', 'workplace']
                if tag in context_tags:
                
                    sentiment=predict_custom_tweet_sentiment(input_msg)
                    print(sentiment)
                    if sentiment>0.5:
                        level='pos'
                      
                    else:

                        if sentiment<=0.1:
                            level='depr'
                           
                        elif sentiment>0.1 and sentiment<=0.3:
                            level='anx'
                        
                        elif sentiment>0.3 and sentiment<=0.5:
                            level='sad'
                        
                    op_tag=tag+'_'+level
                    return random.choice(response_dict[op_tag])
                
                else:
                    #its an info tag
                    print(tag)
                    sentiment=predict_custom_tweet_sentiment(input_msg)
                    print(sentiment)
                    return random.choice(intent['responses'])
    else:
        return "I do not understand..."

  