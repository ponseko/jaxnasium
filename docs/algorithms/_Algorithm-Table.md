
<table>
<thead>
<tr>
<th>Algorithm</th>
<th>Multi-Agent<sup>1</sup></th>
<th>Observation Spaces</th>
<th>Action Spaces</th>
<th>Composite (nested) Spaces<sup>2</sup></th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>PPO</strong></td>
<td>✅</td>
<td><code>Box</code>, <code>Discrete</code>, <code>MultiDiscrete</code></td>
<td><code>Box</code>, <code>Discrete</code>, <code>MultiDiscrete</code></td>
<td>✅</td>
</tr>
<tr>
<td><strong>DQN</strong></td>
<td>✅</td>
<td><code>Box</code>, <code>Discrete</code>, <code>MultiDiscrete</code></td>
<td><code>Discrete</code>, <code>MultiDiscrete</code><sup>3</sup></td>
<td>✅</td>
</tr>
<tr>
<td><strong>PQN</strong></td>
<td>✅</td>
<td><code>Box</code>, <code>Discrete</code>, <code>MultiDiscrete</code></td>
<td><code>Discrete</code>, <code>MultiDiscrete</code><sup>3</sup></td>
<td>✅</td>
</tr>
<tr>
<td><strong>SAC</strong></td>
<td>✅</td>
<td><code>Box</code>, <code>Discrete</code>, <code>MultiDiscrete</code></td>
<td><code>Box</code>, <code>Discrete</code>, <code>MultiDiscrete</code></td>
<td>✅</td>
</tr>
<tr>
<td colspan="5"><em><sup>1</sup> All algorithms support automatic multi-agent transformation through the <code>auto_upgrade_multi_agent</code> parameter</em>. See <a href="../Multi-Agent/">Multi-Agent</a> for more information.</td>
</tr>
<tr>
<td colspan="5"><em><sup>2</sup> Algorithms support composite (nested) spaces. See <a href="../../api/Spaces/">Spaces</a> for more information.</td>
</tr>
<tr>
<td colspan="5"><em><sup>3</sup> MultiDiscrete action spaces in PQN and DQN are only supported when flattening to a Discrete action space. E.g. via the <code>FlattenActionSpaceWrapper</code>.</em></td>
</tr>
</tbody>
</table>