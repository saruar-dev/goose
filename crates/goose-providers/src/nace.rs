use crate::api_client::ApiClient;
use crate::decision::{DecisionProvider, DecisionRequest, DecisionResponse};
use crate::errors::ProviderError;
use crate::http_status::read_json_response;
use crate::openai_compatible::handle_status;
use async_trait::async_trait;
use serde_json::{Map, Value};

pub const NACE_DEFAULT_HOST: &str = "https://console.nace.ai";
pub const NACE_DEFAULT_MODEL: &str = "drex-max";
pub const NACE_KNOWN_MODELS: &[&str] = &["drex-max", "drex-v2.0"];

pub struct NaceProvider {
    api_client: ApiClient,
}

impl NaceProvider {
    pub fn new(api_client: ApiClient) -> Self {
        Self { api_client }
    }
}

#[async_trait]
impl DecisionProvider for NaceProvider {
    async fn create_decision(
        &self,
        request: &DecisionRequest,
    ) -> Result<DecisionResponse, ProviderError> {
        let payload = serde_json::to_value(request).map_err(|error| {
            ProviderError::RequestFailed(format!("Failed to serialize decision request: {error}"))
        })?;
        let response = self
            .api_client
            .request("v1/systemone")
            .response_post(&payload)
            .await?;
        let response = handle_status(response).await?;
        let body: Value = read_json_response(response).await?;
        serde_json::from_value(normalize_response(body)).map_err(|error| {
            ProviderError::RequestFailed(format!(
                "Response body is not a decision response: {error}"
            ))
        })
    }
}

/// Drex returns a score answer's `legend` as the ordered list of level labels and identifies the
/// call with `request_id`; `DecisionResponse` keys the legend by level and calls the id `id`.
fn normalize_response(mut body: Value) -> Value {
    let Some(object) = body.as_object_mut() else {
        return body;
    };
    if let Some(request_id) = object.get("request_id").cloned() {
        object.entry("id").or_insert(request_id);
    }
    let answers = object
        .get_mut("answers")
        .and_then(Value::as_object_mut)
        .into_iter()
        .flat_map(Map::values_mut);
    for answer in answers {
        if let Some(Value::Array(levels)) = answer.get_mut("legend") {
            let keyed: Map<String, Value> = std::mem::take(levels)
                .into_iter()
                .enumerate()
                .map(|(level, label)| (level.to_string(), label))
                .collect();
            answer["legend"] = Value::Object(keyed);
        }
    }
    body
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_client::AuthMethod;
    use crate::decision::{DecisionAnswer, DecisionQuestion};
    use serde_json::json;
    use std::collections::HashMap;
    use wiremock::matchers::{body_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn provider(server: &MockServer) -> NaceProvider {
        NaceProvider::new(
            ApiClient::new_with_tls(server.uri(), AuthMethod::BearerToken("key".into()), None)
                .unwrap(),
        )
    }

    #[tokio::test]
    async fn creates_system_one_decision() {
        let server = MockServer::start().await;
        let request = DecisionRequest {
            model: NACE_DEFAULT_MODEL.to_string(),
            state: json!("Please help ASAP"),
            questions: HashMap::from([(
                "urgent".to_string(),
                DecisionQuestion::Noul {
                    instructions: "Does this express urgency?".to_string(),
                    criteria: None,
                },
            )]),
        };

        Mock::given(method("POST"))
            .and(path("/v1/systemone"))
            .and(body_json(&request))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "model": "drex-max",
                "answers": {"urgent": {"type": "noul", "noul": 0.99}},
                "usage": {"input_tokens": 12, "output_tokens": 3},
                "evaluation_time_ms": 180.8,
                "request_id": "req_123"
            })))
            .expect(1)
            .mount(&server)
            .await;

        let response = provider(&server).create_decision(&request).await.unwrap();

        assert_eq!(
            response.answers["urgent"],
            DecisionAnswer::Noul { noul: 0.99 }
        );
        assert_eq!(response.id.as_deref(), Some("req_123"));
        assert_eq!(response.usage.input_tokens, Some(12));
    }

    #[tokio::test]
    async fn keys_score_legend_by_level() {
        let server = MockServer::start().await;
        let request = DecisionRequest {
            model: NACE_DEFAULT_MODEL.to_string(),
            state: json!("The lamp arrived cracked."),
            questions: HashMap::from([(
                "clarity".to_string(),
                DecisionQuestion::Score {
                    instructions: "How clearly is the problem described?".to_string(),
                    criteria: vec![
                        "unclear".to_string(),
                        "partly clear".to_string(),
                        "clear".to_string(),
                    ],
                },
            )]),
        };

        Mock::given(method("POST"))
            .and(path("/v1/systemone"))
            .and(body_json(&request))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "model": "drex-max",
                "answers": {"clarity": {
                    "type": "score",
                    "score": 0.93,
                    "confidence": 0.705,
                    "legend": ["unclear", "partly clear", "clear"],
                    "probabilities": {"0": 0.133, "1": 0.803, "2": 0.063}
                }},
                "usage": {"input_tokens": 30, "output_tokens": 9},
                "request_id": "req_456"
            })))
            .expect(1)
            .mount(&server)
            .await;

        let response = provider(&server).create_decision(&request).await.unwrap();

        assert_eq!(
            response.answers["clarity"],
            DecisionAnswer::Score {
                score: 0.93,
                confidence: 0.705,
                legend: HashMap::from([
                    ("0".to_string(), json!("unclear")),
                    ("1".to_string(), json!("partly clear")),
                    ("2".to_string(), json!("clear")),
                ]),
                probabilities: HashMap::from([
                    ("0".to_string(), 0.133),
                    ("1".to_string(), 0.803),
                    ("2".to_string(), 0.063),
                ]),
            }
        );
    }
}
