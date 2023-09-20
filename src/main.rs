use std::error::Error;
// 模型
use tch::jit;
use tch::Tensor;
// 分词
use rust_tokenizers::tokenizer::{BertTokenizer, Tokenizer, TruncationStrategy};
use rust_tokenizers::vocab::{Vocab, BertVocab};

// 错误处理
pub type Result<T> = std::result::Result<T, Box<dyn Error>>;


struct RoformerModel {
    model: jit::CModule,
    tokenizer: BertTokenizer,
}

struct RoformerModelInputs {
    inputs: Tensor,
    segment_ids: Tensor,
    attention_mask: Tensor,
}

impl RoformerModel {
    fn new(model_path: String, vocab_path: String) -> Result<Self> {
        let vocab = BertVocab::from_file(&vocab_path)?;
        let tokenizer = BertTokenizer::from_existing_vocab(vocab, true, true);
        // 模型
        let mut model = jit::CModule::load(model_path)?;
        model.set_eval();
        Ok(Self {
            model,
            tokenizer,
        })
    }

    fn infer(&self, text: &String) -> Result<Vec<Vec<f32>>> {
        let token_data = self.tokenizer.encode(text, None, 128, &TruncationStrategy::LongestFirst, 0);
        let inputs = Tensor::from_slice2(&[token_data.token_ids]);

        // 根据模型要求需要改一下数据类型
        let segment_ids = token_data.segment_ids.iter().map(|&x| x as i64).collect::<Vec<i64>>();
        let segment_ids = Tensor::from_slice2(&[segment_ids]);
        let attention_mask = Tensor::from_slice2(&[token_data.special_tokens_mask]).ones_like();

        // let inputs = Tensor::from_slice(&data);
        let outputs_tensor = self.model.forward_ts(&[inputs, attention_mask, segment_ids])?;

        // tensor转换为原生vec
        let outputs = Vec::<Vec<f32>>::try_from(outputs_tensor)?.clone();
        Ok(outputs)
    }

    fn batch_encode(&self, text: Vec<&str>) -> RoformerModelInputs {

        // 获得编码与最大长度
        let token_data = self.tokenizer.encode_list(&text, 1024, &TruncationStrategy::LongestFirst, 0);
        let max_len = &token_data.iter().map(|x| x.token_ids.len()).max().unwrap();

        // 定义两个闭包，补充[PAD]解决对齐问题
        let token_fill_pad = |mut data: Vec<i64>| {
            let len = &data.len();
            let diff = max_len - len;
            match diff {
                0 => data,
                _ => {
                    let mut pad = vec![0i64; max_len - data.len()];
                    pad.push(102);
                    data.truncate(len - 1);
                    data.append(&mut pad);
                    data
                }
            }
        };
        let other_fill_pad = |_data, num| vec![num; *max_len];

        // 修正两个关键参数
        let token_ids_mid = token_data.iter().map(|x| token_fill_pad(x.token_ids.clone())).collect::<Vec<Vec<i64>>>();
        let segment_ids_mid = token_data.iter().map(|x| other_fill_pad(x.segment_ids.clone(), 0i64)).collect::<Vec<Vec<i64>>>();
        let attention_mask_mid = token_data.iter().map(|x| other_fill_pad(x.special_tokens_mask.clone(), 1i64)).collect::<Vec<Vec<i64>>>();

        // 生成tensor
        let inputs = Tensor::from_slice2(&token_ids_mid);
        let segment_ids = Tensor::from_slice2(&segment_ids_mid);
        let attention_mask = Tensor::from_slice2(&attention_mask_mid).ones_like();

        RoformerModelInputs { inputs, segment_ids, attention_mask }
    }

    fn batch_infer(&self, text: Vec<&str>) -> Result<Vec<Vec<f32>>> {
        let data = self.batch_encode(text);
        let outputs_tensor = self.model.forward_ts(&[data.inputs, data.attention_mask, data.segment_ids])?;
        let outputs = Vec::<Vec<f32>>::try_from(outputs_tensor)?.clone();
        Ok(outputs)
    }
}

fn main() -> Result<()> {
    println!("Hello, world!");
    let vocab_path = "data/vocab.txt".to_string();
    let model_path = "data/doc2vec.jit".to_string();
    let roformer_model = RoformerModel::new(model_path, vocab_path)?;
    let a = roformer_model.infer(&"我爱你".to_string())?;
    let b = &a[0][0..3];
    println!("{b:?}", );


    let c = roformer_model.batch_infer(["我爱你", "你爱我", "rust很好YYSD"].to_vec())?;
    println!("{:?}", &c[0][0..3]);
    println!("{:?}", &c[1][0..3]);
    println!("{:?}", &c[0][0..3]);
    Ok(())
}
