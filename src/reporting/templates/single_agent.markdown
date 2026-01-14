# {{ 'title' | translate }} - {{ agent_name }}

**{{ 'generated_at' | translate }}:** {{ generated_at | format_datetime }}

---

## {{ 'evaluation_results' | translate }}

### Основные метрики

| Метрика | Значение |
|---------|----------|
| {{ 'reward' | translate }} ({{ 'mean' | translate }}) | {{ evaluation_results.mean_reward | format_number }} |
| {{ 'reward' | translate }} ({{ 'std' | translate }}) | {{ evaluation_results.std_reward | format_number }} |
| {{ 'episode_length' | translate }} ({{ 'mean' | translate }}) | {{ evaluation_results.mean_episode_length | format_number }} |
| {{ 'episode_length' | translate }} ({{ 'std' | translate }}) | {{ evaluation_results.std_episode_length | format_number }} |
{% if evaluation_results.success_rate is not none %}| {{ 'success_rate' | translate }} | {{ evaluation_results.success_rate | format_percentage }} |{% endif %}
| Количество эпизодов | {{ evaluation_results.num_episodes }} |

{% if statistics and config.include_statistics %}
## {{ 'statistics' | translate }}

{% if statistics.reward %}
### Статистика по наградам

| Показатель | Значение |
|------------|----------|
| {{ 'mean' | translate }} | {{ statistics.reward.mean | format_number }} |
| {{ 'std' | translate }} | {{ statistics.reward.std | format_number }} |
| {{ 'min' | translate }} | {{ statistics.reward.min | format_number }} |
| {{ 'max' | translate }} | {{ statistics.reward.max | format_number }} |
| {{ 'median' | translate }} | {{ statistics.reward.median | format_number }} |
{% endif %}
{% endif %}

{% if quantitative_results %}
## Детальные результаты

- **Количество эпизодов:** {{ evaluation_results.num_episodes }}
{% if quantitative_results.rewards %}- **Записанных наград:** {{ quantitative_results.rewards | length }}{% endif %}
{% if quantitative_results.episode_lengths %}- **Записанных длин эпизодов:** {{ quantitative_results.episode_lengths | length }}{% endif %}

{% endif %}

---

*Отчет сгенерирован системой обучения RL агентов*