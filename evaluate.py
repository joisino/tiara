import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

from wordcloud import WordCloud

from methods import RandomQuery, Tiara, TiaraS, EPSGreedy, UCB
from environments import get_class_ids, get_env
from utils import load_glove


def save_array(opt, budget, env_name, method_name, class_id, seed):
    scores = np.array([opt.history[i][1] for i in range(budget)])
    np.save('outputs/{}_{}_{}_{}_scores.npy'.format(env_name, method_name, class_id, seed), scores)


def update_pics(fig, opt, env, ts, num_methods, method_ind):
    history = [opt.history[i - 1] for i in ts]
    for ind, (loop, score, i) in enumerate(history):
        ax = fig.add_subplot(num_methods, len(ts), len(ts) * method_ind + ind + 1)
        img = env.get_image(i)
        ax.imshow(img)
        ax.text(0, img.size[1] + 100, 'i: {}\ns: {:.4f}\n{}'.format(loop + 1, score, i), size=16, color='red')
        ax.axis('off')


def savefig(fig, basename):
    fig.savefig('outputs/{}.png'.format(basename), bbox_inches='tight')
    fig.savefig('outputs/{}.svg'.format(basename), bbox_inches='tight')


def save_curve(scores, methods, env_name, class_id):
    fig, ax = plt.subplots()
    for method_name, _, _ in methods:
        ax.plot(scores[method_name].mean(0), label=method_name)
    ax.legend()
    fig.savefig('outputs/{}_{}_curve.png'.format(env_name, class_id), bbox_inches='tight')


def wordcloud_col(word, font_size, position, orientation, font_path, random_state):
    lam = (font_size - 6) / (48 - 6)
    red = np.array([255, 75, 0])
    grey = np.array([132, 145, 158])
    res = lam * red + (1 - lam) * grey
    res = res.astype(int)
    return (res[0], res[1], res[2])


def save_wordcloud(opt, env_name, class_id, seed, method_name, font_path):
    tag_scores = opt.tag_scores()
    score_dict = {tag: tag_scores[tag_id] for tag_id, tag in enumerate(opt.tags)}

    x, y = np.ogrid[:300, :300]
    mask = (x - 150) ** 2 + (y - 150) ** 2 > 150 ** 2
    mask = 255 * mask.astype(int)

    wc = WordCloud(font_path=font_path, background_color='white', mask=mask, random_state=0, prefer_horizontal=1.0, max_font_size=48, min_font_size=6)
    wc.generate_from_frequencies(score_dict)
    wc.recolor(random_state=0, color_func=wordcloud_col)

    wc.to_file('outputs/{}_{}_{}_{}_wordcloud.png'.format(env_name, class_id, seed, method_name))

    with open('outputs/{}_{}_{}_{}_wordcloud.svg'.format(env_name, class_id, seed, method_name), 'w') as f:
        f.write(wc.to_svg().replace('fill:(', 'fill:rgb('))


if not os.path.exists('outputs'):
    os.makedirs('outputs')

parser = argparse.ArgumentParser()
parser.add_argument('--tuning', action='store_true')
parser.add_argument('--extra', action='store_true')
parser.add_argument('--env', choices=['open', 'flickr', 'flickrsim'])
parser.add_argument('--num_seeds', type=int, default=10)
parser.add_argument('--budget', type=int, default=500)
parser.add_argument('--api_key', type=str, help='API key for Flickr.')
parser.add_argument('--api_secret', type=str, help='API secret key for Flickr.')
parser.add_argument('--font_path', type=str, help='Font path for wordclouds.')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('-c', '--classes', type=int, nargs='*')
args = parser.parse_args()

glove = load_glove(300, 6)

if args.tuning:
    glove50 = load_glove(50, 6)
    glove100 = load_glove(100, 6)
    glove200 = load_glove(200, 6)

budget = args.budget
budget_ini = 1
class_ids = get_class_ids(args.env)
num_seeds = args.num_seeds
ts = [10, 50, 100, 200, 300, 400, 500]  # checkpoints

print(args.classes)

if args.classes:
    class_ids = [class_ids[c] for c in args.classes]

print('classes:', class_ids)

methods = [
    ('Tiara_1_0.01', Tiara, {'word_embedding': glove, 'lam': 1, 'alpha': 0.01, 'uncase': True}),
    ('UCB_1', UCB, {'alpha': 1.0}),
    ('random', RandomQuery, {})
]

if args.extra:
    methods += [
        ('TiaraS_1_0.01', TiaraS, {'word_embedding': glove, 'lam': 1, 'alpha': 0.01}),
        ('eps_0.01', EPSGreedy, {'eps': 0.01}),
        ('eps_0.1', EPSGreedy, {'eps': 0.1}),
        ('eps_0.5', EPSGreedy, {'eps': 0.5}),
        ('UCB_0.1', UCB, {'alpha': 0.1}),
        ('UCB_10', UCB, {'alpha': 10.0}),
        ('adaeps_0.1', EPSGreedy, {'eps': 0.1, 'adaptive': True}),
        ('adaUCB_1', UCB, {'alpha': 1.0, 'adaptive': True}),
    ]

if args.tuning:
    methods += [
        ('Tiara_1_0.001', Tiara, {'word_embedding': glove, 'lam': 1, 'alpha': 0.001}),
        ('Tiara_1_0.1', Tiara, {'word_embedding': glove, 'lam': 1, 'alpha': 0.1}),
        ('Tiara_1_1', Tiara, {'word_embedding': glove, 'lam': 1, 'alpha': 1}),
        ('Tiara_1_10', Tiara, {'word_embedding': glove, 'lam': 1, 'alpha': 10}),
        ('Tiara_1_100', Tiara, {'word_embedding': glove, 'lam': 1, 'alpha': 100}),
        ('Tiara_0.01_0.01', Tiara, {'word_embedding': glove, 'lam': 0.01, 'alpha': 0.01}),
        ('Tiara_0.1_0.01', Tiara, {'word_embedding': glove, 'lam': 0.1, 'alpha': 0.01}),
        ('Tiara_10_0.01', Tiara, {'word_embedding': glove, 'lam': 10, 'alpha': 0.01}),
        ('Tiara_100_0.01', Tiara, {'word_embedding': glove, 'lam': 100, 'alpha': 0.01}),
        ('Tiara_1000_0.01', Tiara, {'word_embedding': glove, 'lam': 1000, 'alpha': 0.01}),
        ('Tiara_50dim', Tiara, {'word_embedding': glove50, 'lam': 1, 'alpha': 0.01}),
        ('Tiara_100dim', Tiara, {'word_embedding': glove100, 'lam': 1, 'alpha': 0.01}),
        ('Tiara_200dim', Tiara, {'word_embedding': glove200, 'lam': 1, 'alpha': 0.01}),
    ]

for class_ind, class_id in enumerate(class_ids):
    scores = {method_name: np.zeros((num_seeds, budget)) for method_name, _, _ in methods}
    for seed in range(num_seeds):
        fig_pics = plt.figure(figsize=(len(ts) * 4, len(methods) * 3))
        for method_ind, (method_name, Opt, config) in enumerate(methods):
            if args.verbose:
                print(method_name, class_ind, seed)
            env = get_env(args.env, class_id, seed, args.api_key, args.api_secret)
            opt = Opt(env, budget, seed, budget_ini=budget_ini, verbose=args.verbose, **config)
            opt.optimize()
            scores[method_name][seed] = [opt.history[i][1] for i in range(budget)]
            update_pics(fig_pics, opt, env, ts, len(methods), method_ind)
            if hasattr(opt, 'tag_scores'):
                save_wordcloud(opt, args.env, class_id, seed, method_name, args.font_path)
            if hasattr(env, 'save_cache'):
                env.save_cache()
        savefig(fig_pics, '{}_{}_{}_figures'.format(args.env, class_id, seed))
        plt.close()
    save_curve(scores, methods, args.env, class_id)

    for method_name, _, _ in methods:
        np.save('outputs/{}_{}_{}_scores.npy'.format(args.env, class_id, method_name), scores[method_name])
