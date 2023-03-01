from qa_prediction import run


input = [
{
"context": 'ฉันกินมาม่า',
"qas": [
{
"question": "ฉันกินอะไร",
"id": "0",
}
],
}
]

if __name__ == "__main__":
       run(input)
