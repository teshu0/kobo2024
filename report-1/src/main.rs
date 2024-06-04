use candle_core::{Device, Result, Tensor};

pub fn and(a: &Tensor, b: &Tensor, device: &Device) -> Result<Tensor> {
    let inputs = Tensor::cat(&[a, b], 0)?;

    let weights = Tensor::new(&[0.5f32, 0.5f32], &device)?;
    let bias = Tensor::new(&[-0.7f32], &device)?;

    let sum = (weights * &inputs)?.sum(0)?.broadcast_add(&bias)?;

    if sum.get(0)?.to_scalar::<f32>()? <= 0.0f32 {
        Ok(Tensor::new(&[0.0f32], &device)?)
    } else {
        Ok(Tensor::new(&[1.0f32], &device)?)
    }
}

fn or(a: &Tensor, b: &Tensor, device: &Device) -> Result<Tensor> {
    let inputs = Tensor::cat(&[a, b], 0)?;

    let weights = Tensor::new(&[0.5f32, 0.5f32], &device)?;
    let bias = Tensor::new(&[-0.2f32], &device)?;

    let sum = (weights * &inputs)?.sum(0)?.broadcast_add(&bias)?;

    if sum.get(0)?.to_scalar::<f32>()? <= 0.0f32 {
        Ok(Tensor::new(&[0.0f32], &device)?)
    } else {
        Ok(Tensor::new(&[1.0f32], &device)?)
    }
}

fn nand(a: &Tensor, b: &Tensor, device: &Device) -> Result<Tensor> {
    let inputs = Tensor::cat(&[a, b], 0)?;

    let weights = Tensor::new(&[-0.5f32, -0.5f32], &device)?;
    let bias = Tensor::new(&[0.7f32], &device)?;

    let sum = (weights * &inputs)?.sum(0)?.broadcast_add(&bias)?;

    if sum.get(0)?.to_scalar::<f32>()? <= 0.0f32 {
        Ok(Tensor::new(&[0.0f32], &device)?)
    } else {
        Ok(Tensor::new(&[1.0f32], &device)?)
    }
}

fn xor(a: &Tensor, b: &Tensor, device: &Device) -> Result<Tensor> {
    let nand_output = nand(a, b, device)?;
    let or_output = or(a, b, device)?;

    let and_output = and(&nand_output, &or_output, device)?;

    Ok(and_output)
}

fn main() -> Result<()> {
    let device = Device::Cpu;

    let pairs: Vec<[f32; 2]> = vec![[0., 0.], [1., 0.], [0., 1.], [1., 1.]];

    for pair in pairs {
        let a = Tensor::new(&[pair[0]], &device)?;
        let b = Tensor::new(&[pair[1]], &device)?;

        let output = xor(&a, &b, &device)?;

        println!(
            "{} XOR {} => {}",
            pair[0],
            pair[1],
            output.get(0)?.to_scalar::<f32>()?
        );
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;

    // given inputs and expected outputs
    macro_rules! test_layer {
        ($name:ident, $layer:ident, $inputs:expr, $expected:expr) => {
            #[test]
            fn $name() -> Result<()> {
                let device = Device::Cpu;

                let a = Tensor::new(&[$inputs[0] as f32], &device)?;
                let b = Tensor::new(&[$inputs[1] as f32], &device)?;

                let output = $layer(&a, &b, &device)?;

                assert_eq!(output.get(0).unwrap().to_scalar::<f32>()?, $expected);

                Ok(())
            }
        };
    }

    test_layer!(test_and_00, and, [0.0, 0.0], 0.0);
    test_layer!(test_and_01, and, [0.0, 1.0], 0.0);
    test_layer!(test_and_10, and, [1.0, 0.0], 0.0);
    test_layer!(test_and_11, and, [1.0, 1.0], 1.0);

    test_layer!(test_or_00, or, [0.0, 0.0], 0.0);
    test_layer!(test_or_01, or, [0.0, 1.0], 1.0);
    test_layer!(test_or_10, or, [1.0, 0.0], 1.0);
    test_layer!(test_or_11, or, [1.0, 1.0], 1.0);

    test_layer!(test_nand_00, nand, [0.0, 0.0], 1.0);
    test_layer!(test_nand_01, nand, [0.0, 1.0], 1.0);
    test_layer!(test_nand_10, nand, [1.0, 0.0], 1.0);
    test_layer!(test_nand_11, nand, [1.0, 1.0], 0.0);

    test_layer!(test_xor_00, xor, [0.0, 0.0], 0.0);
    test_layer!(test_xor_01, xor, [0.0, 1.0], 1.0);
    test_layer!(test_xor_10, xor, [1.0, 0.0], 1.0);
    test_layer!(test_xor_11, xor, [1.0, 1.0], 0.0);
}
