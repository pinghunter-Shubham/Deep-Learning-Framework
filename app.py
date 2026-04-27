import os, sys, io, csv, json, queue, threading
import numpy as np
from flask import Flask, render_template, request, jsonify, Response, stream_with_context

sys.path.insert(0, os.path.dirname(__file__))
app = Flask(__name__)

# ── Global training state ─────────────────────────────────────────────────────
_queue     = queue.Queue()
_active    = False
_stop_flag = threading.Event()

# ── Custom dataset state ──────────────────────────────────────────────────────
_custom = {
    'X_train': None, 'X_test': None,
    'y_train': None, 'y_test': None,
    'feature_names': [], 'class_names': [],
    'n_features': 0,    'n_classes': 0,
    'norm_mean': None,  'norm_std': None,
}
_trained_model     = None   # kept in memory for inference after training
_trained_data_src  = 'mnist'


# ── Helper ────────────────────────────────────────────────────────────────────

def _put(t, **kw):
    _queue.put({'type': t, **kw})


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


# ─ CSV upload ─────────────────────────────────────────────────────────────────

@app.route('/upload', methods=['POST'])
def upload():
    """Parse an uploaded CSV and return metadata for the UI."""
    global _custom
    try:
        f       = request.files['file']
        content = f.read().decode('utf-8', errors='ignore')
        label_col = int(request.form.get('label_col', -1))

        X, y, feat_names, cls_names, preview = _parse_csv(content, label_col)

        # Train / test split  80 / 20
        n       = len(X)
        idx     = np.random.permutation(n)
        split   = int(n * 0.8)
        tr_idx, te_idx = idx[:split], idx[split:]

        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # Normalise
        mean = X_tr.mean(axis=0)
        std  = X_tr.std(axis=0) + 1e-8
        X_tr = (X_tr - mean) / std
        X_te = (X_te - mean) / std

        _custom.update(dict(
            X_train=X_tr, X_test=X_te,
            y_train=y_tr, y_test=y_te,
            feature_names=feat_names, class_names=cls_names,
            n_features=len(feat_names), n_classes=len(cls_names),
            norm_mean=mean, norm_std=std,
        ))

        # Class distribution
        counts = {cls_names[c]: int(np.sum(y == c)) for c in range(len(cls_names))}

        return jsonify({
            'ok': True,
            'n_samples':    n,
            'n_features':   len(feat_names),
            'n_classes':    len(cls_names),
            'feature_names': feat_names,
            'class_names':  cls_names,
            'class_counts': counts,
            'preview':      preview,
        })
    except Exception as e:
        import traceback
        return jsonify({'ok': False, 'error': str(e), 'trace': traceback.format_exc()}), 400


@app.route('/sample_csv')
def sample_csv():
    """Return a ready-to-download Iris CSV so users can test immediately."""
    rows = [
        'sepal_length,sepal_width,petal_length,petal_width,species',
        '5.1,3.5,1.4,0.2,setosa','4.9,3.0,1.4,0.2,setosa','4.7,3.2,1.3,0.2,setosa',
        '4.6,3.1,1.5,0.2,setosa','5.0,3.6,1.4,0.2,setosa','5.4,3.9,1.7,0.4,setosa',
        '4.6,3.4,1.4,0.3,setosa','5.0,3.4,1.5,0.2,setosa','4.4,2.9,1.4,0.2,setosa',
        '4.9,3.1,1.5,0.1,setosa','5.4,3.7,1.5,0.2,setosa','4.8,3.4,1.6,0.2,setosa',
        '4.8,3.0,1.4,0.1,setosa','4.3,3.0,1.1,0.1,setosa','5.8,4.0,1.2,0.2,setosa',
        '5.7,4.4,1.5,0.4,setosa','5.4,3.9,1.3,0.4,setosa','5.1,3.5,1.4,0.3,setosa',
        '5.7,3.8,1.7,0.3,setosa','5.1,3.8,1.5,0.3,setosa','5.4,3.4,1.7,0.2,setosa',
        '5.1,3.7,1.5,0.4,setosa','4.6,3.6,1.0,0.2,setosa','5.1,3.3,1.7,0.5,setosa',
        '4.8,3.4,1.9,0.2,setosa','5.0,3.0,1.6,0.2,setosa','5.0,3.4,1.6,0.4,setosa',
        '5.2,3.5,1.5,0.2,setosa','5.2,3.4,1.4,0.2,setosa','4.7,3.2,1.6,0.2,setosa',
        '4.8,3.1,1.6,0.2,setosa','5.4,3.4,1.5,0.4,setosa','5.2,4.1,1.5,0.1,setosa',
        '5.5,4.2,1.4,0.2,setosa','4.9,3.1,1.5,0.2,setosa','5.0,3.2,1.2,0.2,setosa',
        '5.5,3.5,1.3,0.2,setosa','4.9,3.6,1.4,0.1,setosa','4.4,3.0,1.3,0.2,setosa',
        '5.1,3.4,1.5,0.2,setosa','5.0,3.5,1.3,0.3,setosa','4.5,2.3,1.3,0.3,setosa',
        '4.4,3.2,1.3,0.2,setosa','5.0,3.5,1.6,0.6,setosa','5.1,3.8,1.9,0.4,setosa',
        '4.8,3.0,1.4,0.3,setosa','5.1,3.8,1.6,0.2,setosa','4.6,3.2,1.4,0.2,setosa',
        '5.3,3.7,1.5,0.2,setosa','5.0,3.3,1.4,0.2,setosa',
        '7.0,3.2,4.7,1.4,versicolor','6.4,3.2,4.5,1.5,versicolor',
        '6.9,3.1,4.9,1.5,versicolor','5.5,2.3,4.0,1.3,versicolor',
        '6.5,2.8,4.6,1.5,versicolor','5.7,2.8,4.5,1.3,versicolor',
        '6.3,3.3,4.7,1.6,versicolor','4.9,2.4,3.3,1.0,versicolor',
        '6.6,2.9,4.6,1.3,versicolor','5.2,2.7,3.9,1.4,versicolor',
        '5.0,2.0,3.5,1.0,versicolor','5.9,3.0,4.2,1.5,versicolor',
        '6.0,2.2,4.0,1.0,versicolor','6.1,2.9,4.7,1.4,versicolor',
        '5.6,2.9,3.6,1.3,versicolor','6.7,3.1,4.4,1.4,versicolor',
        '5.6,3.0,4.5,1.5,versicolor','5.8,2.7,4.1,1.0,versicolor',
        '6.2,2.2,4.5,1.5,versicolor','5.6,2.5,3.9,1.1,versicolor',
        '5.9,3.2,4.8,1.8,versicolor','6.1,2.8,4.0,1.3,versicolor',
        '6.3,2.5,4.9,1.5,versicolor','6.1,2.8,4.7,1.2,versicolor',
        '6.4,2.9,4.3,1.3,versicolor','6.6,3.0,4.4,1.4,versicolor',
        '6.8,2.8,4.8,1.4,versicolor','6.7,3.0,5.0,1.7,versicolor',
        '6.0,2.9,4.5,1.5,versicolor','5.7,2.6,3.5,1.0,versicolor',
        '5.5,2.4,3.8,1.1,versicolor','5.5,2.4,3.7,1.0,versicolor',
        '5.8,2.7,3.9,1.2,versicolor','6.0,2.7,5.1,1.6,versicolor',
        '5.4,3.0,4.5,1.5,versicolor','6.0,3.4,4.5,1.6,versicolor',
        '6.7,3.1,4.7,1.5,versicolor','6.3,2.3,4.4,1.3,versicolor',
        '5.6,3.0,4.1,1.3,versicolor','5.5,2.5,4.0,1.3,versicolor',
        '5.5,2.6,4.4,1.2,versicolor','6.1,3.0,4.6,1.4,versicolor',
        '5.8,2.6,4.0,1.2,versicolor','5.0,2.3,3.3,1.0,versicolor',
        '5.6,2.7,4.2,1.3,versicolor','5.7,3.0,4.2,1.2,versicolor',
        '5.7,2.9,4.2,1.3,versicolor','6.2,2.9,4.3,1.3,versicolor',
        '5.1,2.5,3.0,1.1,versicolor','5.7,2.8,4.1,1.3,versicolor',
        '6.3,3.3,6.0,2.5,virginica','5.8,2.7,5.1,1.9,virginica',
        '7.1,3.0,5.9,2.1,virginica','6.3,2.9,5.6,1.8,virginica',
        '6.5,3.0,5.8,2.2,virginica','7.6,3.0,6.6,2.1,virginica',
        '4.9,2.5,4.5,1.7,virginica','7.3,2.9,6.3,1.8,virginica',
        '6.7,2.5,5.8,1.8,virginica','7.2,3.6,6.1,2.5,virginica',
        '6.5,3.2,5.1,2.0,virginica','6.4,2.7,5.3,1.9,virginica',
        '6.8,3.0,5.5,2.1,virginica','5.7,2.5,5.0,2.0,virginica',
        '5.8,2.8,5.1,2.4,virginica','6.4,3.2,5.3,2.3,virginica',
        '6.5,3.0,5.5,1.8,virginica','7.7,3.8,6.7,2.2,virginica',
        '7.7,2.6,6.9,2.3,virginica','6.0,2.2,5.0,1.5,virginica',
        '6.9,3.2,5.7,2.3,virginica','5.6,2.8,4.9,2.0,virginica',
        '7.7,2.8,6.7,2.0,virginica','6.3,2.7,4.9,1.8,virginica',
        '6.7,3.3,5.7,2.1,virginica','7.2,3.2,6.0,1.8,virginica',
        '6.2,2.8,4.8,1.8,virginica','6.1,3.0,4.9,1.8,virginica',
        '6.4,2.8,5.6,2.1,virginica','7.2,3.0,5.8,1.6,virginica',
        '7.4,2.8,6.1,1.9,virginica','7.9,3.8,6.4,2.0,virginica',
        '6.4,2.8,5.6,2.2,virginica','6.3,2.8,5.1,1.5,virginica',
        '6.1,2.6,5.6,1.4,virginica','7.7,3.0,6.1,2.3,virginica',
        '6.3,3.4,5.6,2.4,virginica','6.4,3.1,5.5,1.8,virginica',
        '6.0,3.0,4.8,1.8,virginica','6.9,3.1,5.4,2.1,virginica',
        '6.7,3.1,5.6,2.4,virginica','6.9,3.1,5.1,2.3,virginica',
        '5.8,2.7,5.1,1.9,virginica','6.8,3.2,5.9,2.3,virginica',
        '6.7,3.3,5.7,2.5,virginica','6.7,3.0,5.2,2.3,virginica',
        '6.3,2.5,5.0,1.9,virginica','6.5,3.0,5.2,2.0,virginica',
        '6.2,3.4,5.4,2.3,virginica','5.9,3.0,5.1,1.8,virginica',
    ]
    from flask import make_response
    resp = make_response('\n'.join(rows))
    resp.headers['Content-Type']        = 'text/csv'
    resp.headers['Content-Disposition'] = 'attachment; filename=iris_sample.csv'
    return resp


# ─ Prediction on custom data ──────────────────────────────────────────────────

@app.route('/predict_custom', methods=['POST'])
def predict_custom():
    global _trained_model, _trained_data_src
    if _trained_model is None:
        return jsonify({'error': 'No trained model. Train first.'}), 400
    try:
        vals = request.get_json()['features']        # list of floats
        X    = np.array(vals, dtype=np.float32).reshape(1, -1)

        if _trained_data_src == 'custom' and _custom['norm_mean'] is not None:
            X = (X - _custom['norm_mean']) / _custom['norm_std']

        from minitorch.tensor import Tensor
        _trained_model.eval()
        logits = _trained_model(Tensor(X))
        raw    = logits.data[0]
        exp_v  = np.exp(raw - raw.max())
        probs  = (exp_v / exp_v.sum()).tolist()
        pred   = int(np.argmax(probs))

        cls_names = (
            _custom['class_names'] if _trained_data_src == 'custom'
            else [str(i) for i in range(10)]
        )
        return jsonify({
            'prediction': pred,
            'class_name': cls_names[pred],
            'probabilities': [round(p, 4) for p in probs],
            'class_names':   cls_names,
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


# ─ Train ──────────────────────────────────────────────────────────────────────

@app.route('/train', methods=['POST'])
def train():
    global _active, _stop_flag, _queue
    if _active:
        return jsonify({'error': 'Training already running'}), 409
    cfg        = request.get_json()
    _queue     = queue.Queue()
    _stop_flag = threading.Event()
    _active    = True
    threading.Thread(target=_run_training, args=(cfg,), daemon=True).start()
    return jsonify({'status': 'started'})


@app.route('/stop', methods=['POST'])
def stop():
    _stop_flag.set()
    return jsonify({'status': 'stopping'})


@app.route('/stream')
def stream():
    def generate():
        while True:
            try:
                msg = _queue.get(timeout=25)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get('type') in ('done', 'error', 'stopped'):
                    break
            except queue.Empty:
                yield 'data: {"type":"ping"}\n\n'
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


# ── Training worker ───────────────────────────────────────────────────────────

def _run_training(cfg):
    global _active, _trained_model, _trained_data_src
    try:
        from minitorch.tensor import Tensor
        import minitorch.nn as nn
        import minitorch.optim as optim

        data_src  = cfg.get('data_source', 'mnist')
        lr        = float(cfg.get('lr', 0.001))
        epochs    = int(cfg.get('epochs', 3))
        bs        = int(cfg.get('batch_size', 128))
        opt_name  = cfg.get('optimizer', 'adam')
        model_type = cfg.get('model', 'mlp')

        # ── Load data ─────────────────────────────────────────────────────
        if data_src == 'mnist':
            from minitorch.dataset import load_mnist
            _put('status', message='Loading MNIST dataset…')
            X_tr, Y_tr = load_mnist('training', 'data')
            X_te, Y_te = load_mnist('testing',  'data')
            X_tr = X_tr.astype(np.float32) / 255.0
            X_te = X_te.astype(np.float32) / 255.0
            n_classes = 10
            class_names = [str(i) for i in range(10)]

            def one_hot(y):
                oh = np.zeros((len(y), n_classes), dtype=np.float32)
                oh[np.arange(len(y)), y] = 1.0
                return oh
            Y_tr_oh = one_hot(Y_tr)

        else:  # custom CSV
            if _custom['X_train'] is None:
                _put('error', message='No dataset uploaded. Please upload a CSV first.')
                return
            X_tr    = _custom['X_train']
            X_te    = _custom['X_test']
            Y_tr    = _custom['y_train']
            Y_te    = _custom['y_test']
            n_classes   = _custom['n_classes']
            class_names = _custom['class_names']

            def one_hot(y):
                oh = np.zeros((len(y), n_classes), dtype=np.float32)
                oh[np.arange(len(y)), y] = 1.0
                return oh
            Y_tr_oh = one_hot(Y_tr)

        # ── Build model ───────────────────────────────────────────────────
        if data_src == 'mnist':
            if model_type == 'cnn':
                model = nn.Sequential(
                    nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
                    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
                    nn.Flatten(), nn.Linear(32 * 7 * 7, 10),
                )
            else:
                model = nn.Sequential(
                    nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(256, 128), nn.ReLU(),
                    nn.Linear(128, 10),
                )
        else:
            # Auto-size MLP for custom data
            nf = _custom['n_features']
            h1 = max(64, min(256, nf * 8))
            h2 = max(32, h1 // 2)
            model = nn.Sequential(
                nn.Linear(nf, h1), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(h1, h2), nn.ReLU(),
                nn.Linear(h2, n_classes),
            )

        optimizer = (
            optim.Adam(model.parameters(), lr=lr)
            if opt_name == 'adam'
            else optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        )
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.StepLR(optimizer, step_size=1, gamma=0.5)

        n_batches   = max(1, len(X_tr) // bs)
        total_steps = epochs * n_batches

        _put('config',
             model=model_type, optimizer=opt_name, lr=lr,
             epochs=epochs, batch_size=bs, n_batches=n_batches,
             data_source=data_src,
             n_features=_custom['n_features'] if data_src == 'custom' else 784,
             n_classes=n_classes, class_names=class_names)

        step          = 0
        loss_history  = []
        epoch_metrics = []

        for epoch in range(epochs):
            if _stop_flag.is_set():
                _put('stopped', message='Training stopped by user.'); return

            model.train()
            idx        = np.random.permutation(len(X_tr))
            ep_loss    = 0.0
            ep_correct = 0

            for i in range(n_batches):
                if _stop_flag.is_set():
                    _put('stopped', message='Training stopped by user.'); return

                batch = idx[i * bs:(i + 1) * bs]

                if data_src == 'mnist' and model_type == 'cnn':
                    xb = Tensor(X_tr[batch].reshape(-1, 1, 28, 28))
                else:
                    xb = Tensor(X_tr[batch])
                yb = Tensor(Y_tr_oh[batch])

                optimizer.zero_grad()
                logits = model(xb)
                loss   = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                lv     = float(loss.data.item())
                preds  = np.argmax(logits.data, axis=1)
                acc    = float(np.mean(preds == Y_tr[batch]))
                ep_loss    += lv
                ep_correct += int(np.sum(preds == Y_tr[batch]))
                step += 1

                if i % max(1, n_batches // 20) == 0 or i == n_batches - 1:
                    loss_history.append(round(lv, 4))
                    _put('batch',
                         epoch=epoch + 1, batch=i + 1, n_batches=n_batches,
                         loss=round(lv, 4), acc=round(acc, 4),
                         progress=round(step / total_steps, 4),
                         loss_history=loss_history[-100:])

            # ── Epoch eval ─────────────────────────────────────────────────
            model.eval()
            if data_src == 'mnist' and model_type == 'cnn':
                test_preds = _eval_batched(model, X_te, cnn=True)
            else:
                test_preds = _eval_batched(model, X_te, cnn=False)

            test_acc  = float(np.mean(test_preds == Y_te))
            train_acc = ep_correct / len(X_tr)
            avg_loss  = ep_loss / n_batches
            scheduler.step()

            epoch_metrics.append({
                'epoch': epoch + 1,
                'train_acc': round(train_acc, 4),
                'test_acc':  round(test_acc, 4),
                'avg_loss':  round(avg_loss, 4),
            })
            _put('epoch', **epoch_metrics[-1], all_epochs=epoch_metrics)

        # ── Save trained model ─────────────────────────────────────────────
        _trained_model    = model
        _trained_data_src = data_src

        # ── Sample predictions ─────────────────────────────────────────────
        model.eval()
        if data_src == 'mnist':
            from minitorch.dataset import load_mnist
            _, Y_te_raw = load_mnist('testing', 'data')
            sample_idx = np.random.choice(len(X_te), 10, replace=False)
            inp = Tensor(
                X_te[sample_idx].reshape(-1, 1, 28, 28)
                if model_type == 'cnn' else X_te[sample_idx]
            )
            logits_s  = model(inp)
            sp_preds  = np.argmax(logits_s.data, axis=1).tolist()
            sp_labels = Y_te[sample_idx].tolist()
            sp_images = (X_te[sample_idx] * 255).astype(np.uint8).tolist()
            _put('done',
                 data_source='mnist',
                 predictions=sp_preds, labels=sp_labels, images=sp_images,
                 final_test_acc=round(test_acc, 4),
                 epoch_metrics=epoch_metrics,
                 class_names=class_names)
        else:
            sample_idx = np.random.choice(len(X_te), min(10, len(X_te)), replace=False)
            inp        = Tensor(X_te[sample_idx])
            logits_s   = model(inp)
            sp_preds   = np.argmax(logits_s.data, axis=1).tolist()
            sp_labels  = Y_te[sample_idx].tolist()
            # send un-normalised feature values for display
            raw_x      = (X_te[sample_idx] * _custom['norm_std'] + _custom['norm_mean']).tolist()
            _put('done',
                 data_source='custom',
                 predictions=sp_preds, labels=sp_labels, raw_x=raw_x,
                 final_test_acc=round(test_acc, 4),
                 epoch_metrics=epoch_metrics,
                 feature_names=_custom['feature_names'],
                 class_names=class_names)

    except Exception as exc:
        import traceback as tb
        _put('error', message=str(exc), traceback=tb.format_exc())
    finally:
        _active = False


def _eval_batched(model, X, cnn=False, chunk=512):
    from minitorch.tensor import Tensor
    preds = []
    for j in range(0, len(X), chunk):
        xc  = X[j:j + chunk]
        inp = Tensor(xc.reshape(-1, 1, 28, 28) if cnn else xc)
        preds.append(np.argmax(model(inp).data, axis=1))
    return np.concatenate(preds)


# ── CSV parser ────────────────────────────────────────────────────────────────

def _parse_csv(content, label_col_idx=-1):
    reader  = csv.reader(io.StringIO(content))
    rows    = [r for r in reader if any(c.strip() for c in r)]
    if not rows:
        raise ValueError('Empty file')

    headers = [h.strip() for h in rows[0]]
    n_cols  = len(headers)

    if label_col_idx < 0:
        label_col_idx = n_cols + label_col_idx   # e.g. -1 → last

    feat_cols = [i for i in range(n_cols) if i != label_col_idx]

    X_rows, y_raw = [], []
    for row in rows[1:]:
        row = [c.strip() for c in row]
        if len(row) != n_cols:
            continue
        try:
            X_rows.append([float(row[i]) for i in feat_cols])
            y_raw.append(row[label_col_idx])
        except ValueError:
            continue

    if not X_rows:
        raise ValueError('No numeric rows found. Check the file format.')

    X = np.array(X_rows, dtype=np.float32)

    # Encode labels (handles both numeric and string)
    try:
        y_num    = [int(float(v)) for v in y_raw]
        unique   = sorted(set(y_num))
        c_map    = {c: i for i, c in enumerate(unique)}
        y        = np.array([c_map[v] for v in y_num], dtype=int)
        cls_names = [str(c) for c in unique]
    except ValueError:
        unique    = sorted(set(y_raw))
        c_map     = {c: i for i, c in enumerate(unique)}
        y         = np.array([c_map[v] for v in y_raw], dtype=int)
        cls_names = list(unique)

    feat_names = [headers[i] for i in feat_cols]

    # Build a small preview (first 5 rows, original values)
    preview_rows = rows[1:6]
    preview = {
        'headers': headers,
        'rows':    [[c.strip() for c in r] for r in preview_rows if len(r) == n_cols],
    }
    return X, y, feat_names, cls_names, preview


if __name__ == '__main__':
    print('\n  MiniTorch Dashboard → http://127.0.0.1:5000\n')
    app.run(debug=False, threaded=True, port=5000)
